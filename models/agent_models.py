from typing import List, Optional
from typing import TypedDict, Annotated, Any
from enum import Enum

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from pydantic import BaseModel, Field

class Step(BaseModel):
    worker_name: str = Field(
        description="Name of the worker"
    )
    worker_instruction: str = Field(
        description="Complete instruction for the worker to work independently"
    )
    reasoning: str = Field(
        description="Reasoning for why this step is needed and what it should accomplish"
    )

class Plan(BaseModel):
    plan_reasoning: str =  Field(
        description="Reasoning for why this plan is needed and what it should accomplish"
    )
    steps: List[Step]


class TripType(str, Enum):
    """Enum for trip type categories"""
    WEEKEND_TRIP = "weekend_trip"
    WITHIN_COUNTRY_TRIP = "within_country_trip"
    ACROSS_COUNTRY_TRIP = "across_country_trip"

class TripPurpose(str, Enum):
    """Enum for trip purpose categories"""
    HOLY_TRIP = "holy_trip"
    FAMILY_TRIP = "family_trip"
    ADVENTURE = "adventure"
    RELAXATION = "relaxation"
    BUSINESS_TRIP = "business_trip"
    CULTURAL = "cultural"
    HONEYMOON = "honeymoon"
    SOLO_TRAVEL = "solo_travel"
    GROUP_TRAVEL = "group_travel"
    FOOD_TOURISM = "food_tourism"
    WILDLIFE = "wildlife"
    SHOPPING = "shopping"
    EDUCATIONAL = "educational"
    BAG_PACK = "bag_pack"
    OTHER = "other"

class QueryType(str, Enum):
    """Enum for query type classification to determine required entities"""
    TRIP_PLANNING = "trip_planning"  # Requires all 7 entities: destination, origin_city, travel_dates, trip_duration, number_of_travelers, preferences, budget
    INFORMATIONAL = "informational"  # Requires only relevant entities mentioned in query (e.g., "best places to visit in December")
    SPECIFIC_SERVICE = "specific_service"  # Requires context-specific entities (e.g., "hotels in Goa", "flights to Dubai")

class TravelIntent(BaseModel):
    reasoning: str
    query_type: QueryType = Field(
        description="Type of query: trip_planning (requires all 7 entities), informational (requires only relevant entities), or specific_service (requires context-specific entities)"
    )
    trip_type: Optional[TripType] = Field(
        default=None,
        description="Type of trip: weekend_trip, within_country_trip, across_country_trip, or None if not applicable"
    )
    trip_purpose: Optional[TripPurpose] = Field(
        default=None,
        description="Purpose of trip: holy_trip, family_trip, adventure, relaxation, business_trip, cultural, honeymoon, solo_travel, group_travel, food_tourism, wildlife, shopping, educational, bag_pack, other, or None if not applicable"
    )

class TravelEntity(BaseModel):
    """Model to capture key travel entities extracted from user query"""
    reasoning: str = Field(
        description="Reasoning for the entities identified and extraction decisions"
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin/source location of the trip (city, country, or region)"
    )
    destination: Optional[str] = Field(
        default=None,
        description="Destination location(s) for the trip (city, country, or region)"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date of travel in ISO format (YYYY-MM-DD) or natural language if specific date not provided"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date of travel in ISO format (YYYY-MM-DD) or natural language if specific date not provided"
    )
    duration: Optional[str] = Field(
        default=None,
        description="Duration of the trip (e.g., '3 days', '1 week', '2 weeks')"
    )
    number_of_travelers: Optional[int] = Field(
        default=None,
        description="Number of people traveling"
    )
    budget: Optional[str] = Field(
        default=None,
        description="Budget information if mentioned (e.g., 'budget-friendly', '$2000', 'luxury')"
    )
    accommodation_preference: Optional[str] = Field(
        default=None,
        description="Accommodation preferences (e.g., 'hotel', 'hostel', 'resort', 'airbnb')"
    )
    activities: Optional[List[str]] = Field(
        default=None,
        description="List of activities or interests mentioned (e.g., ['hiking', 'sightseeing', 'beach'])"
    )
    special_requirements: Optional[str] = Field(
        default=None,
        description="Any special requirements or constraints (e.g., 'vegetarian food', 'wheelchair accessible', 'pet-friendly')"
    )

class NextStep(BaseModel):
    """
    Model for decision node to determine next step in the workflow.
    
    Attributes:
        user_confirmation: Whether user confirmed to proceed to next step
        next_agent: Name of the next agent to route to if confirmed
        reasoning: Explanation of the decision
    """
    user_confirmation: bool = False
    reasoning: str = ""
    reply_to_user: str = ""

class UserIntent(BaseModel):
    inquiry_type: str
    reply_to_user: str
    is_followup: bool
    regenerated_question: str
    reasoning: str = Field(
        description="Reasoning for the classification and decisions made"
    )

class TravelInquiry(BaseModel):
    reply_to_user: str
    regenerated_question: str
    reasoning: str = Field(
        description="Reasoning for the classification and decisions made"
    )
    number_of_travelers: Optional[int] = Field(
        default=None,
        description="Number of travelers extracted from conversation"
    )
    budget: Optional[str] = Field(
        default=None,
        description="Budget for the trip (total or per-person)"
    )
    preference: Optional[str] = Field(
        default=None,
        description="User preferences for the trip experience"
    )

class TransportDetails(BaseModel):
    reasoning: str = Field(
        default="",
        description="Reasoning for suggested transport"
    )
    recommended_transport_details: Optional[str] = None
    reply_to_user: str = ""

class AccommodationDetails(BaseModel):
    reasoning: str = Field(
        default="",
        description="Reasoning for suggested accommodation"
    )
    recommended_accommodation_details: Optional[str] = None
    reply_to_user: str = ""

class ItineraryDetails(BaseModel):
    reasoning: str = Field(
        description="Reasoning for suggested itinerary"
    )
    recommended_itinerary_details: Optional[str]
    reply_to_user: str

class EmailContent(BaseModel):
    subject: str = ""
    html_content: str = ""
    reply_to_user: str = ""
    email_sent: bool = False

def last_value_reducer(left: Any, right: Any) -> Any:
    """Reducer that returns the last (most recent) value, allowing multiple updates per step."""
    return right

class TravelAgentState(TypedDict):
    user_query: str
    is_followup: bool
    regenerated_question: str
    current_state: str
    query_type: str
    trip_type: str
    trip_purpose: str
    number_of_travelers: Optional[int]
    budget: Optional[str]
    preference: Optional[str]
    responses: Optional[dict]  # Dict with keys: transport_details, accommodation_details, itinerary_details
    user_feedback: Annotated[str, last_value_reducer]
    reply_to_user: str
    email_id: Optional[str]
    subject: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]