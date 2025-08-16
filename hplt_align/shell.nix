{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShellNoCC {
  nativeBuildInputs = with pkgs.buildPackages; [
    (python3.withPackages (ps: with ps; [
      ipywidgets
      jupyterlab
    ]))
  ];

  shellHook = "jupyter lab --no-browser";
}
