import re
from typing import Text, List, Any, Dict

from rasa_sdk import FormValidationAction, Tracker
from rasa_sdk.executor import CollectingDispatcher

valid_month = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
               'september': 9, 'october': 10, 'november': 11,
               'december': 12, 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11,
               'dec': 12}


class ValidateTravelInfo(FormValidationAction):
    def name(self) -> Text:
        return "validate_travel_info_form"

    def validate_month(self,
                       slot_value: Any,
                       dispatcher: CollectingDispatcher,
                       tracker: Tracker,
                       domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print(f"Validating month {slot_value}")
        if slot_value.lower() in valid_month.keys():
            return {"month": slot_value.lower()}
        else:
            dispatcher.utter_message(text=f'Please check the month or please rephrase it ?')
            return {"month": None}

    def validate_days(self,
                      slot_value: Any,
                      dispatcher: CollectingDispatcher,
                      tracker: Tracker,
                      domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        regex = r'^[0-9]+$'
        print(f"Validating days {slot_value}")
        if re.fullmatch(regex, slot_value):
            return {"days": slot_value}
        dispatcher.utter_message(
            text='Please specify number of days in numbers correctly'
        )
        return {"days": None}

    def validate_place(self,
                       slot_value: Any,
                       dispatcher: CollectingDispatcher,
                       tracker: Tracker,
                       domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print(f"Validating place {slot_value}")
        if slot_value and slot_value.replace(" ", "").isalpha():
            print(f"place {slot_value}")
            return {"place": slot_value}

        return {"place": 'World'}
