import re
from typing import Text, Any, Dict

from rasa_sdk import Tracker, ValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

valid_month = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
               'september': 9, 'october': 10, 'november': 11,
               'december': 12, 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11,
               'dec': 12}


class ValidateTravelInfoSlots(ValidationAction):
    def validate_month(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate month value."""
        print(f"Validating month {slot_value}")
        #print(f"User Input {tracker.latest_message}")
        if slot_value.lower() in valid_month.keys():
            return {"month": slot_value.lower()}
        else:
            dispatcher.utter_message(text=f'Please check the month or please rephrase it ?')
            return {"month": None}

    def validate_days(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate days value."""
        print(f"Validating days {slot_value}")
        #print(f"User Input {tracker.latest_message}")
        regex = r'^[0-9]+$'
        if re.fullmatch(regex, slot_value):
            return {"days": slot_value}
        else:
            dispatcher.utter_message(text='Please specify number of days in numbers correctly')
            return {"days": None}

    def validate_destination(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate place value."""
        print(f"Validating place {slot_value}")
        #print(f"User Input {tracker.latest_message}")
        if slot_value and slot_value.replace(" ", "").isalpha():
            return {"destination": slot_value}
        else:
            return {"destination": 'World'}

    def validate_source(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate place value."""
        print(f"Validating place {slot_value}")
        #print(f"User Input {tracker.latest_message}")
        if slot_value and slot_value.replace(" ", "").isalpha():
            return {"source": slot_value}
        else:
            dispatcher.utter_message(text='Please specify desired starting point correctly')
            return {"source": None}

