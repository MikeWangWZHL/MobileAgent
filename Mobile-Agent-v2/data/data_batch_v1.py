from scenarios import Scenario

# restaurant recommendation
scenario_1_groups_1 = {
  "length": 5,
  "scenario": Scenario.RESTAURANT_RECOMMENDATION.name,
  "scenario_id": Scenario.RESTAURANT_RECOMMENDATION.value,
  "tasks": [
      {
          "task_id": "1_late_night_korean_food",
          "instruction": "Find the best-rated late-night Korean restaurant in Champaign, IL that opens beyond 9pm on Google Maps.",
          "type": "single_app",
          "apps": ["Maps"],
      },
      {
          "task_id": "1_nearest_bakery",
          "instruction": "Get directions to the nearest Bakery that has a rating higher than 4.0 on Google Maps. Stop at the screen showing the route.",
          "type": "single_app",
          "apps": ["Maps"]
      },
      {
          "task_id": "1_thai_duck",
          "instruction": "Find the best-rated Thai restaurant in Urbana, IL that serves duck cuisine on Google Maps. Review customer comments and compile a summary of positive and negative feedback in Notes.",
          "type": "multi_app",
          "apps": ["Maps", "Notes"]
      },
      {
          "task_id": "1_bakery_birthday_cake",
          "instruction": "Find a Bakery near me and does birthday cakes on Google Maps. Find the phone number and create a new note in Notes for that.",
          "type": "multi_app",
          "apps": ["Maps", "Notes"]
      },
      {
          "task_id": "1_chinese_ohare",
          "instruction": "Find me a popular Chinese restaurant near Chicago O'Hare airport on Google Maps. Check X for recent posts about their signature dishes and write a summary in Notes. Then get directions to that restaurant on Google Maps. Stop at the screen showing the route.",
          "type": "multi_app",
          "apps": ["Maps", "X", "Notes"]
      }
  ]
}


# information researching
scenario_2_groups_1 = {
    "length": 5,
    "scenario": Scenario.INFORMATION_RESEARCHING.name,
    "scenario_id": Scenario.INFORMATION_RESEARCHING.value,
    "tasks": [
        {
            "task_id": "2_segment_anything_cited",
            "instruction": "Find the most-cited paper that cites the paper 'Segment Anything' on Google Scholar. Stop at the screen showing the paper abstract.",
            "type": "single_app",
            "apps": ["Chrome"]
        },
        {
            "task_id": "2_llm_agents_survey",
            "instruction": "Find at least three representative survey papers on LLM agents on Google Scholar, and add their titles to the Notes.",
            "type": "multi_app",
            "apps": ["Chrome", "Notes"]
        },
        {
            "task_id": "2_headphones_reviews",
            "instruction": "Find three detailed user reviews of the Bose QC45 headphones from Amazon. Summarize the general sentiment in the Notes.",
            "type": "multi_app",
            "apps": ["Amazon", "Notes"]
        },
        {
            "task_id": "2_recipes_chinese",
            "instruction": "I have some onions, beef, and potatoes in my refrigerator. Can you find me a Chinese-style recipe that uses all three ingredients and can be prepared in under an hour? And find me a video tutorial on YouTube for that. Stop at the screen displaying the video.",
            "type": "multi_app",
            "apps": ["Chrome", "YouTube"]
        },
        {
            "task_id": "2_mcdonalds_deals",
            "instruction": "Can you check the MacDonald's APP to see if there are any Rewards or Deals including Spicy McCrispy. If so, help me add that to Mobile Order (Do not pay yet, I will do it myself). And then check the pickup location and get directions on Google Maps. Stop at the screen showing the route.",
            "type": "multi_app",
            "apps": ["McDonald's", "Maps"]
        }
    ]
}


# deal hunting
scenario_3_groups_1 = {
    "length": 5,
    "scenario": Scenario.ONLINE_SHOPPING.name,
    "scenario_id": Scenario.ONLINE_SHOPPING.value,
    "tasks": [
        {
            "task_id": "3_oled_tv",
            "instruction": "Find the best deal on a 55-inch 4K OLED TV at Best Buy. Stop at the screen displaying the best deal you find.",
            "type": "single_app",
            "apps": ["Best Buy"]
        },
        {
            "task_id": "3_laptop_nvidia_gpu",
            "instruction": "Find me a laptop on Amazon that is under $1000 with an Nvidia GPU and more than 8GB RAM.",
            "type": "single_app",
            "apps": ["Amazon Shopping"]
        },
        {
            "task_id": "3_ninja_air_fryer",
            "instruction": "Compare the price of a Ninja air fryer 8 qt at Walmart and Amazon. Stop at the screen displaying the best deal you find.",
            "type": "multi_app",
            "apps": ["Amazon Shopping", "Walmart"]
        },
        {
            "task_id": "3_walmart_sale_items",
            "instruction": "Check if any of the following items are on sale at Walmart: ribeye steak, fresh oranges, or toilet paper. If any are on sale, add a note in Notes with their prices.",
            "type": "multi_app",
            "apps": ["Walmart", "Notes"]
        },
        {
            "task_id": "3_nintendo_switch_joy_con",
            "instruction": "I want to buy a brand-new Nintendo Switch Joy-Con. Any color is fine. Please compare the prices on Amazon, Walmart, and Best Buy. Find the cheapest option and stop at the screen where I can add it to the cart.",
            "type": "multi_app",
            "apps": ["Amazon Shopping", "Best Buy", "Walmart"]
        }
    ]
}

# what's trending: fun stuff on the internet; social media; movies; news
scenario_4_groups_1 = {
    "length": 5,
    "scenario": Scenario.WHATS_TRENDING.name,
    "scenario_id": Scenario.WHATS_TRENDING.value,
    "tasks": [
        {
            "task_id": "4_x_back_myth_wukong",
            "instruction": "Find the top posts about the game 'Black Myth Wukong' on \"X\" and summarize the key highlights in Notes.",
            "type": "multi_app",
            "apps": ["X", "Notes"]
        },
        {
            "task_id": "4_x_trending_news",
            "instruction": "Check the top 3 trending news on \"X\". Read a few posts to figure out what's happening. And create a new Note to summarize your findings.",
            "type": "multi_app",
            "apps": ["X", "Notes"]
        },
        {
            "task_id": "4_watercolor_painting_tutorial",
            "instruction": "I want to learn how to paint watercolor. Find me some content creators to follow on Lemon8 that has highly liked posts about watercolor painting tutorials. List their account names in Notes.",
            "type": "multi_app",
            "apps": ["Lemon8", "Notes"]
        },
        {
            "task_id": "4_movie_trending",
            "instruction": "Check the top 5 trending movies on Fandango that are currently in theaters. Compare their ratings and create a note in Notes for the highest-rated one, including its name and showtimes.",
            "type": "single_app",
            "apps": ["Fandango", "Notes"]
        },
        {
            "task_id": "4_horror_movie_reviews",
            "instruction": "Find me the latest horror movie currently in theaters on Fandango. Check some reviews on Lemon8 about the movie and create a note in Notes with the general sentiment.",
            "type": "multi_app",
            "apps": ["Fandango", "Lemon8" "Notes"]
        }
    ]
}


# travel planning: itinerary planning; hotel recommendations; flight booking, etc
scenario_5_groups_1 = {
  "length": 5,
  "scenario": Scenario.TRAVEL_PLANNING.name,
  "scenario_id": Scenario.TRAVEL_PLANNING.value,
  "tasks": [
        {
            "task_id": "5_cheap_flights_newyork",
            "instruction": "Find the cheapest round-trip flight from Chicago to New York City in the next month on Booking. Stop at the screen showing the best deal.",
            "type": "single_app",
            "apps": ["Booking"]
        },
        {
            "task_id": "5_things_to_do_la",
            "instruction": "Suggest some interesting things to do in LA. Find the top 3 attractions on Tripadvisor. Save the list in Notes.",
            "type": "multi_app",
            "apps": ["Tripadvisor", "Notes"]
        },
        {
            "task_id": "5_palo_alto_tour",
            "instruction": "Plan a one-day itinerary for Palo Alto, CA using Tripadvisor. Choose the attractions and dining recommendations, but keep in mind that I don't like seafood and I love museums. Write the plan in Notes.",
            "type": "multi_app",
            "apps": ["Tripadvisor", "Notes"]
        },
        {
            "task_id": "5_local_food_chicago",
            "instruction": "Find a highly recommended local restaurant in Chicago on Tripadvisor. Check the reviews about must-try dishes and summarize in Notes.",
            "type": "multi_app",
            "apps": ["Tripadvisor", "Notes"]
        },
        {
            "task_id": "5_hotel_champaign",
            "instruction": "Help me find a hotel in Champaign, IL on Booking that is under $200 for a queen bed. Make sure that the rating is higher than 7.0. Double check on Google Maps to see if it is close to the Green Street. Show me your final choice on Booking.",
            "type": "multi_app",
            "apps": ["Booking", "Maps"]
        }
  ]
}

import json
if __name__ == "__main__":
  # batch v1
  data_dir = "/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/Mobile-Agent-v2/data/batch_v1"

  # 1. restaurant recommendation
  output_path = f"{data_dir}/scenario_{Scenario.RESTAURANT_RECOMMENDATION.value}_batch_v1.json"
  with open(output_path, 'w') as f:
    json.dump(scenario_1_groups_1, f, indent=4)

  # 2. information researching
  output_path = f"{data_dir}/scenario_{Scenario.INFORMATION_RESEARCHING.value}_batch_v1.json"
  with open(output_path, 'w') as f:
    json.dump(scenario_2_groups_1, f, indent=4)
  
  # 3. deal hunting
  output_path = f"{data_dir}/scenario_{Scenario.ONLINE_SHOPPING.value}_batch_v1.json"
  with open(output_path, 'w') as f:
    json.dump(scenario_3_groups_1, f, indent=4)
  
  # 4. what's trending
  output_path = f"{data_dir}/scenario_{Scenario.WHATS_TRENDING.value}_batch_v1.json"
  with open(output_path, 'w') as f:
    json.dump(scenario_4_groups_1, f, indent=4)
  
  # 5. travel planning
  output_path = f"{data_dir}/scenario_{Scenario.TRAVEL_PLANNING.value}_batch_v1.json"
  with open(output_path, 'w') as f:
    json.dump(scenario_5_groups_1, f, indent=4)


