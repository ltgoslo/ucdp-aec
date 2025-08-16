"""
Configuration using command line arguments and python files.

Configuration for t5 is first expressed by a schema, declaring options that are needed to describe the model.
These options need to be assigned a value using command line arguments.
Configuration values can be given as direct variable assignement or through a python module.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import pathlib
import types
from typing import Any, Iterator


logger: logging.Logger = logging.getLogger(__name__)


class Config(dict[str, Any]):
    """
    A set of configuration values.

    Config objects are dictionaries that can be access through `var.key` instead of `var["key"]`.
    This class also defines functions for processing command-line arguments.

    Attributes
    ----------
    _root: list[str]
        The location of the config root, empty for the root config.
    _lazy_values: dict[str, str]
        Values that have not been parsed yet because their type is not yet known (because the corresponding key was not declared).
    _schema: dict[str, type]
        Type constraints for configuration keys.
    _help: dict[str, str]
        Short message describing the purpose of each key.
    _default: dict[str, Any]
        Default value for configuration keys, a default does not always need to be declared.
    _from_environment: set[str]
        Set of keys that where imported from the environment instead of command line arguments.

    Config
    ------
    DATA_PATH: pathlib.Path
        Path to the directory containing the datasets.
        This is an environment variable that can be set by exporting `DATA_PATH`.
    LOG_PATH: pathlib.Path
        Path to where logs, metrics and metadata are saved.
        This is an environment variable that can be set by exporting `LOG_PATH`.
    MODEL_PATH: pathlib.Path
        Path to where models are saved.
        This is an environment variable that can be set by exporting `MODEL_PATH`.
    """

    class _LazyMarkerType:
        """Type used to mark lazy values in Config dictionaries."""

    def __init__(self: Config, argv: list[str] | None = None, root: list[str] | None = None, lazy_values: dict[str, str] | None = None) -> None:
        """
        Initialize a configuration object with the given argument list.

        Parameters
        ----------
        argv: list[str], optional
            Command line arguments to extract values from.
            The first element is assumed to be the program name and is ignored.
        root: list[str], optional
            Location of this `Config` object inside the configuration tree.
            For example the `Config` with root `["model", "generator"]` would contains values such as `model.generator.temperature`.
            It is empty for the root object.
            If the root starts with `_test_`, environment variables config keys will not be imported, this is to simplify unit tests.
        lazy_values: dict[str, str], optional
            Config values as dictionary of strings, useful when loading from json.

        """
        super().__init__()
        self._root: list[str] = root if root is not None else []
        self._lazy_values: dict[str, str] = {}
        self._schema: dict[str, type] = {}
        self._help: dict[str, str] = {}
        self._default: dict[str, Any] = {}
        self._from_environment: set[str] = set()
        if argv is not None and lazy_values is not None:
            msg: str = "both argv and lazy_values given"
            raise ValueError(msg)
        if argv is not None:
            for argument in argv[1:]:
                self.process_argument(argument)
        if lazy_values is not None:
            self._lazy_values.update(lazy_values)

        # Do not require the environment variables to be set when running unit tests.
        if len(self._root) == 0 or self._root[0] != "_test_":
            self._import_environment("DATA_PATH", pathlib.Path, "Path to the directory containing the datasets.")
            self._import_environment("LOG_PATH", pathlib.Path, "Path to where logs, metrics and metadata are saved.")
            self._import_environment("MODEL_PATH", pathlib.Path, "Path to where models are saved.")

    def __getattr__(self: Config, name: str) -> Any:
        """Access config values as attributes."""
        if name in self.__dict__:
            return self.__dict__[name]
        if name not in self:
            raise AttributeError(self.key_path(name))
        return self[name]

    def pretty_string(self: Config) -> str:
        """Return a colorful representation of the Config."""
        return "\n".join(self._pretty_color_lines())

    def _pretty_color_lines(self: Config, prefix: str = "") -> Iterator[str]:
        """Describe the Config state as a text lines."""
        reset_color: str = "\033[0m"
        root_color: str = "\033[91m"
        key_color: str = "\033[92m"
        value_color: str = "\033[94m"
        help_color: str = "\033[2m"

        root: str = ".".join(self._root) + " " if self._root else ""
        yield f"{prefix}{root_color}{root}Config{reset_color}"
        for key in sorted(self.keys()):
            value: Any = self[key]
            if isinstance(value, self._LazyMarkerType):
                yield f"{prefix}    {key_color}{key}{reset_color} {help_color}(LAZY){reset_color} = {value_color}{self._lazy_values[key]}{reset_color}"
            elif isinstance(value, Config):
                subconfig: Iterator[str] = value._pretty_color_lines(prefix="    ")
                if self._help.get(key):
                    yield f"{prefix}{next(subconfig)} {help_color}({self._help[key]}){reset_color}"
                yield from (f"{prefix}{line}" for line in subconfig)
            else:
                type_name: str = self._schema[key].__name__ if key in self._schema else "UNDECLARED"
                default_msg: str = f" default: {self._default[key]}" if key in self.default else ""
                help_msg: str = f" {help_color}({self._help[key]}){default_msg}{reset_color}" if self._help.get(key) else ""
                yield f"{prefix}    {key_color}{key}{reset_color} {help_color}({type_name}){reset_color} = {value_color}{value}{reset_color}{help_msg}"

    def _import_environment(self: Config, name: str, cast: type, help_msg: str) -> None:
        """Import an environment variable into the config namespace."""
        try:
            self[name] = cast(os.environ[f"{name}"])
            self._schema[name] = cast
            self._help[name] = help_msg
            self._from_environment.add(name)
        except AttributeError as exception:
            exception.add_note(f"You need to set the environment variable {name}.")
            exception.add_note(f"{name}: {help_msg}")
            raise

    def process_argument(self: Config, argument: str) -> None:
        """
        Process a command line argument.

        Three types of argument are accepted:
            - Arguments starting with a `--` are untyped *lazy* arguments, that will be converted to the correct type once it is declared.
            - Arguments starting with a `@` are eval arguments, they are executed as python code where other configuration keys can be directly accessed and modified as python variables.
            - Other arguments are import arguments, they are treated as paths to python module files.
        """
        if argument.startswith("--"):
            self._process_lazy_argument(argument[2:])
        elif argument.startswith("@"):
            self._process_eval_argument(argument[1:])
        else:
            self._process_import_argument(argument)

    def _resolve_config_path(self: Config, key: str) -> tuple[Config, str]:
        """Resolve a path to the corresponding config object and key."""
        components: list[str] = key.split(".")
        config: Config = self
        for i, component in enumerate(components[:-1]):
            if component not in config:
                config[component] = Config(root=[*self._root, *components[:i+1]])
            config = config[component]
        return config, components[-1]

    @staticmethod
    def _split_assignment(argument: str) -> tuple[str, str | bool]:
        """
        Split an assignment into key and values.

        The expected format is:
            key(=value)?
        If no value is provided, the key is assumed to be a boolean and True is considered to be its implicit value.
        """
        operands: list[str] = argument.split("=", maxsplit=1)
        if len(operands) == 1:
            return argument, True
        return operands[0], operands[1]

    def _convert_dicts(self: Config) -> None:
        """Convert dictionaries to Config objects."""
        for key, value in self.items():
            if isinstance(value, Config):
                value._convert_dicts()
            elif isinstance(value, dict):
                subconfig = Config(root=[*self._root, key])
                for subkey, subvalue in value.items():
                    subconfig[subkey] = subvalue
                subconfig._convert_dicts()
                self[key] = subconfig

    def key_path(self: Config, key: str) -> str:
        """Return the full path of a config key."""
        return ".".join([*self._root, key])

    def _process_lazy_argument(self: Config, argument: str) -> None:
        """Process an assignment with a yet unknown type."""
        lhs: str
        value: str | bool
        lhs, value = self._split_assignment(argument)

        config: Config
        key: str
        config, key = self._resolve_config_path(lhs)
        if isinstance(value, bool):
            config[key] = True
        else:
            config[key] = self._LazyMarkerType()
            config._lazy_values[key] = value

    def _process_eval_argument(self: Config, argument: str) -> None:
        """
        Evaluate a python argument on the config dictionary.

        When passing a string argument through the shell, it must be enclosed in quote (like all python string), which usually need to be escaped.
        """
        exec(argument, self)
        self.pop("__builtins__", None)
        self._convert_dicts()

    def _process_import_argument(self: Config, argument: str) -> None:
        """
        Load python file argument.

        The file is loaded in an independent context, all the variable defined in the file (even through import) are added to the configuration, with the exception of builtins and whole modules.
        """
        spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location("argument", argument)
        if spec is None or spec.loader is None:
            raise ImportError(argument)
        module: types.ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        key: str
        value: Any
        for key, value in vars(module).items():
            if (
                    key not in module.__builtins__
                    and not key.startswith("_")
                    and not isinstance(value, types.ModuleType)
                ):
                self[key] = value
        self._convert_dicts()

    def _resolve_lazy(self: Config, key: str) -> None:
        """Cast lazily typed value to schema type."""
        if self._schema[key] in [str, int, float, bool, pathlib.Path]:
            try:
                self[key] = self._schema[key](self._lazy_values[key])
                self._lazy_values.pop(key)
            except ValueError as exception:
                raise InvalidValueError(self.key_path(key), self._schema[key], self._help[key], self[key]) from exception
        else:
            raise NotImplementedError(f"Lazy {self._schema[key]} config")

    def declare_option(self: Config, full_key: str, type_constraint: type, help_msg: str, default: None | Any = None) -> None:
        """
        Add an option to the configuration schema.

        Parameters
        ----------
        full_key: str
            Configuration key to declare such as `model.generator.temperature`.
            It is preferable to directly declare each option in its direct parent's config, so more often than not, this should be a dot-less key.
        type_constraint: type
            What is the type of the configuration key being declared.
        help_msg: str
            An explanation of what the configuration key is for.
        default: type_constraint, optional
            The value the configuration key will take if it is not assigned a value.
            If a default is not given and the option was not defined by the command-line, this function will raise an exception.

        Raises
        ------
        MissingValueError
            If the command-line arguments do not define the declared option when no default is provided.
        MistypedValueError
            The configuration key was defined by the command-line arguments with a different type from the one being declared.
        InvalidValueError
            The configuration key was lazily defined by the command-line argument but could not be converted to the declared type.

        """
        config: Config
        key: str
        config, key = self._resolve_config_path(full_key)

        config._schema[key] = type_constraint
        config._help[key] = help_msg
        if default is not None:
            config._default[key] = default

        if key not in config:
            if default is None:
                raise MissingValueError(config.key_path(key), type_constraint, help_msg)
            config[key] = default

        if isinstance(config[key], self._LazyMarkerType):
            config._resolve_lazy(key)
        if not isinstance(config[key], type_constraint):
            raise MistypedValueError(config.key_path(key), type_constraint, help_msg, config[key])

    def check_schema(self: Config) -> None:
        """Check whether the config adhere to the schema."""
        for key, value in self.items():
            if key not in self._schema:
                raise UndeclaredValueError(self.key_path(key), value)
            if isinstance(value, Config):
                value.check_schema()

    def log(self: Config) -> None:
        """Write the config to log file."""
        for key in sorted(self):
            value: Any = self[key]
            if isinstance(value, Config):
                value.log()
            else:
                logger.info("value %s %s", self.key_path(key), value)

    def save(self: Config, output_path: pathlib.Path) -> None:
        """Save the compiled configuration to a python file."""
        imports: set[str]
        code: list[str]
        imports, code = self._python_representation()
        with output_path.open("w") as output:
            print("\n".join(imports), file=output)
            print("\n\n", end="", file=output)
            print("\n".join(code), file=output)

    def _python_representation(self: Config) -> tuple[set[str], list[str]]:
        """Get a representation of the configuration as python code."""
        imports: set[str] = set()
        code: list[str] = []
        for key in sorted(self):
            if key in self._from_environment:
                continue
            value: Any = self[key]
            full_key: str = key
            if self._root:
                indices: str = "".join(f"[\"{part}\"]" for part in [*self._root[1:], key])
                full_key = f"{self._root[0]}{indices}"

            if isinstance(value, Config):
                code.append(f"{full_key} = dict()")
                sub_imports: set[str]
                sub_code: list[str]
                sub_imports, sub_code = value._python_representation()
                imports.update(sub_imports)
                code.extend(sub_code)
            else:
                if isinstance(value, pathlib.PosixPath):
                    imports.add("from pathlib import PosixPath")
                code.append(f"{full_key} = {value!r}")
        return imports, code


class ConfigError(Exception):
    """Base Exception class for t5.config."""


class MissingValueError(ConfigError):
    """Exception raised when a needed config value is not provided as command line argument."""

    def __init__(self: MissingValueError, key: str, type_constraint: type, help_msg: str) -> None:
        """Initiate a MissingValueError with given meta information."""
        super().__init__(f"Missing configuration value for: {key}\nType constraint: {type_constraint}\nHelp: {help_msg}")


class MistypedValueError(ConfigError):
    """Exception raised when a config value is given the wrong type."""

    def __init__(self: MistypedValueError, key: str, type_constraint: type, help_msg: str, value: Any) -> None:
        """Initiate a MistypedValueError with given meta information and wrong value."""
        super().__init__(f"Wrong type for configuration value: {key}\nType constraint: {type_constraint}\nActual type was {type(value)} from value {value}\nHelp: {help_msg}")


class InvalidValueError(ConfigError):
    """Exception raised when a config value cannot be converted to the correct type."""

    def __init__(self: InvalidValueError, key: str, type_constraint: type, help_msg: str, value: Any) -> None:
        """Initiate an InvalidValueError with given meta information and wrong value."""
        super().__init__(f"Cannot convert configuration value: {key}\nType constraint: {type_constraint}\nAssigned value: {value}\nHelp: {help_msg}")


class UndeclaredValueError(ConfigError):
    """Exception raised when a config value is given but not declared by the program."""

    def __init__(self: UndeclaredValueError, key: str, value: Any) -> None:
        """Initiate an UndeclaredValueError with given meta information."""
        super().__init__(f"Extra configuration value was provided but it is undeclared: {key}\nAssigned value: {value}")
