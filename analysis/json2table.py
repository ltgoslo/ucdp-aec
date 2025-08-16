import json


MAIN_ORDER = [
    "actor accuracy", "date accuracy", "location accuracy", "deaths accuracy", "aggregate accuracy",
    "actor semantic p@1", "location semantic p@1",
]


APPENDIX_ORDER = [
    "side_a_name accuracy",
    "side_b_name accuracy",
    "start_date accuracy",
    "end_date accuracy",
    "location_root_name accuracy",
    "location_adm1_name accuracy",
    "location_adm2_name accuracy",
    "location_where_name accuracy",
    "deaths_side_a accuracy",
    "deaths_side_b accuracy",
    "deaths_civilian accuracy",
    "deaths_unknown accuracy",
    "deaths_low accuracy",
    "deaths_high accuracy",
]


def format_key(results, key):
    if "p@1" in key:
        return f"{100*results[key]:.1f}"
    else:
        return f"{results[key]:.1f}"


def main() -> None:
    results = json.loads(input())
    main = " & ".join(format_key(results, key) for key in MAIN_ORDER)
    appendix = " & ".join(format_key(results, key) for key in APPENDIX_ORDER)
    print(f"MAIN TABLE & {main} \\\\")
    print(f"APPENDIX TABLE & {appendix} \\\\")


if __name__ == "__main__":
    main()
