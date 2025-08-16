import numpy
import re
import sys


INPUT_STYLE_SET = ['event_type', 'event_type_sent', 'keywords', 'triggers', 'template']
OUTPUT_STYLE_SET = ['argument:sentence']

BREAK_PENALTY = 10  # Penalty if variable is not preceded by space or followed by space/period.
TEMPLATE_MARKER = '\1'  # Template marker for variables
POST_TEMPLATE_VALID = " .,"  # Unpenalized characters after template variable
PRE_TEMPLATE_VALID = " "  # Unpenalized characters before template variable

UCDP_TEMPLATE_FORMAT = """The conflict between {side_a_name} (A) and {side_b_name} (B) resulted in causalities between {start_date} and {end_date} at {location_where_name} in {location_adm2_name} in the region of {location_adm1_name} in {location_root_name}. {deaths_side_a} people died on side A, {deaths_side_b} people died on side B, {deaths_civilian} civilians died and {deaths_unknown} unidentified people died. The low estimate for the total number of people that died is {deaths_low}. The high estimate for the total number of people that died is {deaths_high}."""

UCDP_FIELDS = re.findall(r"{([^}]+)}", UCDP_TEMPLATE_FORMAT)
UCDP_TEMPLATE = re.sub(" {[^}]*}", TEMPLATE_MARKER, UCDP_TEMPLATE_FORMAT)

ROLE_TO_TYPE = {
	 "side_a_name": "Actor",
	 "side_b_name": "Actor",
	 "start_date": "Date",
	 "end_date": "Date",
	 "location_root_name": "Place",
	 "location_adm1_name": "Place",
	 "location_adm2_name": "Place",
	 "location_where_name": "Place",
	 "deaths_side_a": "Deaths",
	 "deaths_side_b": "Deaths",
	 "deaths_civilian": "Deaths",
	 "deaths_unknown": "Deaths",
	 "deaths_low": "Deaths",
	 "deaths_high": "Deaths",
}

TYPE_PH_MAP = {
    "Actor": "somebody",
    "Date": "sometime",
    "Place": "somewhere",
    "Deaths": "some number",
}


def fuzzy_align(prediction):
    """
    Use a Levenshtein algorithm to fuzzily align the output of a model with its template.

    As currently written, the variables must always be preceded by a space in the template.
    Furthermore, the following character is expected to be one of POST_TEMPLATE_VALID.
    In any case, when matching, a PRE_TEMPLATE_VALID will be enforced before a variable and any POST_TEMPLATE_VALID character will match the template just after a variable.
    """
    m = numpy.empty((len(prediction)+1, len(UCDP_TEMPLATE)+1), dtype=numpy.int64)
    parents = numpy.zeros((len(prediction)+1, len(UCDP_TEMPLATE)+1), dtype=numpy.int8)

    m[:, 0] = numpy.arange(len(prediction)+1)
    parents[1:, 0] = 1

    template_init = (numpy.array(list(UCDP_TEMPLATE))==TEMPLATE_MARKER)*(BREAK_PENALTY-1)+1
    m[0, 1:] = template_init.cumsum()
    parents[0, 1:] = 2


    for y in range(1, len(prediction)+1):
        for x in range(1, len(UCDP_TEMPLATE)+1):
            if x>1 and UCDP_TEMPLATE[x-2] == TEMPLATE_MARKER:
                addition = 1
                deletion = BREAK_PENALTY
                transformation = (0 if prediction[y-1] in POST_TEMPLATE_VALID else BREAK_PENALTY)
            elif UCDP_TEMPLATE[x-1] == TEMPLATE_MARKER:
                addition = 0
                deletion = BREAK_PENALTY
                transformation = (0 if prediction[y-1] in PRE_TEMPLATE_VALID else BREAK_PENALTY)
            else:
                addition = 1
                deletion = 1
                transformation = (UCDP_TEMPLATE[x-1] != prediction[y-1])

            m[y][x] = min(m[y-1][x] + addition, m[y][x-1] + deletion, m[y-1][x-1] + transformation)
            if m[y-1][x] + addition == m[y][x]:
                parents[y][x] |= 1
            if m[y][x-1] + deletion == m[y][x]:
                parents[y][x] |= 2
            if m[y-1][x-1] + transformation == m[y][x]:
                parents[y][x] |= 4

    matches = {location+1: [] for location, value in enumerate(UCDP_TEMPLATE) if value == TEMPLATE_MARKER}
    y, x = len(prediction), len(UCDP_TEMPLATE)
    while (y, x) != (0, 0):
        if UCDP_TEMPLATE[x-1] == TEMPLATE_MARKER:
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

    return { field: "".join(match[::-1]).strip() for field, (_, match) in zip(UCDP_FIELDS, sorted(matches.items())) }


class eve_template_generator():
    def __init__(self, passage, triggers, roles, input_style, output_style, vocab, instance_base=False):
        """
        generate strctured information for events
        
        args:
            passage(List): a list of tokens
            triggers(List): a list of triggers
            roles(List): a list of Roles
            input_style(List): List of elements; elements belongs to INPUT_STYLE_SET
            input_style(List): List of elements; elements belongs to OUTPUT_STYLE_SET
            instance_base(Bool): if instance_base, we generate only one pair (use for trigger generation), else, we generate trigger_base (use for argument generation)
        """
        self.raw_passage = passage
        self.triggers = triggers
        self.roles = roles
        self.events = self.process_events(passage, triggers, roles)
        self.input_style = input_style
        self.output_style = output_style
        self.vocab = vocab
        self.event_templates = []
        if instance_base:
            for e_type in self.vocab['event_type_itos']:
                theclass = getattr(sys.modules[__name__], e_type.replace(':', '_').replace('-', '_'), False)
                if theclass:
                    self.event_templates.append(theclass(self.input_style, self.output_style, passage, e_type, self.events))
                else:
                    print(e_type)

        else:
            for event in self.events:
                theclass = getattr(sys.modules[__name__], event['event type'].replace(':', '_').replace('-', '_'), False)
                assert theclass
                self.event_templates.append(theclass(self.input_style, self.output_style, event['tokens'], event['event type'], event))
        self.data = [x.generate_pair(x.trigger_text) for x in self.event_templates]
        self.data = [x for x in self.data if x]

    def get_training_data(self):
        return self.data

    def process_events(self, passage, triggers, roles):
        """
        Given a list of token and event annotation, return a list of structured event

        structured_event:
        {
            'trigger text': str,
            'trigger span': (start, end),
            'event type': EVENT_TYPE(str),
            'arguments':{
                ROLE_TYPE(str):[{
                    'argument text': str,
                    'argument span': (start, end)
                }],
                ROLE_TYPE(str):...,
                ROLE_TYPE(str):....
            }
            'passage': PASSAGE
        }
        """
        
        events = {trigger: [] for trigger in triggers}

        for argument in roles:
            trigger = argument[0]
            events[trigger].append(argument)
        
        event_structures = []
        for trigger, arguments in events.items():
            eve_type = trigger[2]
            eve_text = "UNK"
            eve_span = (trigger[0], trigger[1])
            argus = {}
            for argument in arguments:
                role_type = argument[1][2]
                if role_type not in argus.keys():
                    argus[role_type] = []
                argus[role_type].append({
                    'argument text': argument[1][3],
                    'argument span': (argument[1][0], argument[1][1]),
                })
            event_structures.append({
                'trigger text': eve_text,
                'trigger span': eve_span,
                'event type': eve_type,
                'arguments': argus,
                'passage': ' '.join(passage),
                'tokens': passage
            })
        return event_structures

class event_template():
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        self.input_style = input_style
        self.output_style = output_style
        self.output_template = self.get_output_template()
        self.passage = ' '.join(passage)
        self.tokens = passage
        self.event_type = event_type
        if gold_event is not None:
            self.gold_event = gold_event
            if isinstance(gold_event, list):
                # instance base
                self.trigger_text = " and ".join([x['trigger text'] for x in gold_event if x['event type']==event_type])
                self.trigger_span = [x['trigger span'] for x in gold_event if x['event type']==event_type]
                self.arguments = [x['arguments'] for x in gold_event if x['event type']==event_type]
            else:
                # trigger base
                self.trigger_text = gold_event['trigger text']
                self.trigger_span = [gold_event['trigger span']]
                self.arguments = [gold_event['arguments']]         
        else:
            self.gold_event = None
        
    @classmethod
    def get_keywords(self):
        pass

    def generate_pair(self, query_trigger):
        """
        Generate model input sentence and output sentence pair
        """
        input_str = self.generate_input_str(query_trigger)
        output_str, gold_sample = self.generate_output_str(query_trigger)
        return (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens)

    def generate_input_str(self, query_trigger):
        return None

    def generate_output_str(self, query_trigger):
        return (None, False)

    def decode(self, prediction):
        pass

    def evaluate(self, predict_output):
        assert self.gold_event is not None
        converted_gold = self.get_converted_gold()
        correct = 0
        total = 0
        per_type = { rtype: [] for rtype in ROLE_TO_TYPE.values() }
        for role, argument in converted_gold.items():
            total += 1
            value = (argument == predict_output.get(role))
            correct += value
            per_type[ROLE_TO_TYPE[role]].append(value)
        return {
            'gold_tri_num': 0, 
            'pred_tri_num': 0,
            'match_tri_num': 0,
            'gold_arg_num': len(converted_gold),
            'pred_arg_num': len(predict_output),
            'match_arg_id': correct,
            'match_arg_cls': correct,
            'aec_agg': sum(sum(values)/len(values) for values in per_type.values())/len(per_type)
        }
    
    def get_converted_gold(self):
        converted_gold = {}
        for argu in self.arguments:
            for arg_type, arg_list in argu.items():
                for arg in arg_list:
                    converted_gold[arg_type] = arg['argument text'].strip()
        return converted_gold


class UCDPDeath(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['die', 'kill']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format(UCDP_TEMPLATE.format(**{field: TYPE_PH_MAP[role] for field, role in ROLE_TO_TYPE.items()}))
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('death in conflict event')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to conflict where someone is being killed.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = {
                            # In our case, the role should always be in argu
                            role: argu[role][0]["argument text"] if role in argu else TYPE_PH_MAP[rtype]
                            for role, rtype in ROLE_TO_TYPE.items()
                        }
                        output_texts.append(UCDP_TEMPLATE.format(**filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))

        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        return fuzzy_align(preds)
