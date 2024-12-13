### information_research_and_summarization ###
from enum import Enum
class Scenario(Enum):
    RESTAURANT_RECOMMENDATION = 1
    INFORMATION_RESEARCHING = 2
    ONLINE_SHOPPING = 3
    WHATS_TRENDING = 4
    TRAVEL_PLANNING = 5

# print(Scenario.RESTAURANT_RECOMMENDATION.name)
# print(Scenario.RESTAURANT_RECOMMENDATION.value)
# print(Scenario(1).name)
# print(Scenario['RESTAURANT_RECOMMENDATION'].value)