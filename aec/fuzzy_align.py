import re
import numpy


BREAK_PENALTY = 10  # Penalty if variable is not preceded by space or followed by space/period.
TEMPLATE_MARKER = '\1'  # Template marker for variables
POST_TEMPLATE_VALID = " .,"  # Unpenalized characters after template variable
PRE_TEMPLATE_VALID = " "  # Unpenalized characters before template variable


class FuzzyAlign:
    """
    Use a Levenshtein algorithm to fuzzily align the output of a model with its template.

    As currently written, the variables must always be preceded by a space in the template.
    Furthermore, the following character is expected to be one of POST_TEMPLATE_VALID.
    In any case, when matching, a PRE_TEMPLATE_VALID will be enforced before a variable and any POST_TEMPLATE_VALID character will match the template just after a variable.
    """
    def __init__(self, template):
        self.fields = re.findall(r"{([^}]+)}", template)
        self.template = re.sub(" {[^}]*}", TEMPLATE_MARKER, template)

    def __call__(self, prediction):
        m = numpy.empty((len(prediction)+1, len(self.template)+1), dtype=numpy.int64)
        parents = numpy.zeros((len(prediction)+1, len(self.template)+1), dtype=numpy.int8)

        m[:, 0] = numpy.arange(len(prediction)+1)

        template_init = (numpy.array(list(self.template))==TEMPLATE_MARKER)*(BREAK_PENALTY-1)+1
        m[0, 1:] = template_init.cumsum()


        for y in range(1, len(prediction)+1):
            for x in range(1, len(self.template)+1):
                if x>1 and self.template[x-2] == TEMPLATE_MARKER:
                    addition = 1
                    deletion = BREAK_PENALTY
                    transformation = (0 if prediction[y-1] in POST_TEMPLATE_VALID else BREAK_PENALTY)
                elif self.template[x-1] == TEMPLATE_MARKER:
                    addition = 0
                    deletion = BREAK_PENALTY
                    transformation = (0 if prediction[y-1] in PRE_TEMPLATE_VALID else BREAK_PENALTY)
                else:
                    addition = 1
                    deletion = 1
                    transformation = (self.template[x-1] != prediction[y-1])

                m[y][x] = min(m[y-1][x] + addition, m[y][x-1] + deletion, m[y-1][x-1] + transformation)
                if m[y-1][x] + addition == m[y][x]:
                    parents[y][x] |= 1
                if m[y][x-1] + deletion == m[y][x]:
                    parents[y][x] |= 2
                if m[y-1][x-1] + transformation == m[y][x]:
                    parents[y][x] |= 4

        matches = {location+1: [] for location, value in enumerate(self.template) if value == TEMPLATE_MARKER}
        y, x = len(prediction), len(self.template)
        while (y, x) != (0, 0):
            if self.template[x-1] == TEMPLATE_MARKER:
                if parents[y][x] & 1:
                    y -= 1
                    matches[x].append(prediction[y])
                elif parents[y][x] & 4:
                    y -= 1
                    x -= 1
                elif parents[y][x] & 2:
                    x -= 1
            else:
                if parents[y][x] & 4:
                    y -= 1
                    x -= 1
                elif parents[y][x] & 1:
                    y -= 1
                elif parents[y][x] & 2:
                    x -= 1

        return { field: "".join(match[::-1]).strip() for field, (_, match) in zip(self.fields, sorted(matches.items())) }


if __name__ == "__main__":
    prediction = "The conflict between Government of Syria (A) and IS (B) resulted in causalities between 2020-02-22 and 2020-01-22 at Al Baghouz town in Albu Kamal district in the region of Deir ez Zor governorate in Syria. 0 people died on side A, 1 people died in side B, 0 civilians died and 0 unidentified people died. The low estimate for the total number of people that died is 1. The high estimate for this total number is 1 people."
    template = "The conflict between {side_a_name} (A) and {side_b_name} (B) resulted in causalities between {start_date} and {end_date} at {location_where_name} in {location_adm2_name} in the region of {location_adm1_name} in {location_root_name}. {deaths_side_a} people died on side A, {deaths_side_b} people died on side B, {deaths_civilian} civilians died and {deaths_unknown} unidentified people died. The low estimate for the total number of people that died is {deaths_low}. The high estimate for the total number of people that died is {deaths_high}."

    print("\n".join(f"{key}: |{value}|" for key, value in FuzzyAlign(template)(prediction).items()))
