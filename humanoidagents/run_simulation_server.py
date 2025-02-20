import json
import os
import argparse
from datetime import datetime, timedelta
import logging
import random
from collections import defaultdict
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

from humanoid_agent import HumanoidAgent
from location import Location
from utils import DatetimeNL, load_json_file, write_json_file, bucket_agents_by_location, override_agent_kwargs_with_condition, get_curr_time_to_daily_event

import requests
from flask import Flask, request, url_for, request, jsonify
from flask_caching import Cache


# server side caching
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)

@app.route('/')
def index():
    return 'hi'

@app.route('/chat_single_turn', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def chat_single_turn_with_possible_responses(n_responses=3):
    data = parse_request(request, expected_keys=['curr_date', 'specific_time', 'initiator_name', 'responder_name', 'conversation_history'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    initiator_name = data['initiator_name']
    responder_name = data['responder_name']
    conversation_history = data['conversation_history']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

    initiator = name_to_agent[initiator_name]
    responder = name_to_agent[responder_name]

    possible_responses = []
    for i in range(n_responses):
        reaction = responder.get_agent_reaction_about_another_agent(initiator, curr_time, conversation_history=conversation_history)
        response = responder.speak_to_other_agent(initiator, curr_time, reaction=reaction, conversation_history=conversation_history)
        possible_responses.append(response)

    return possible_responses

#need variable number of messages thorugh POST + give users options + user defines what to use 
@app.route('/chat', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def chat():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time', 'initiator_name', 'responder_name'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    initiator_name = data['initiator_name']
    responder_name = data['responder_name']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

    initiator = name_to_agent[initiator_name]
    responder = name_to_agent[responder_name]
    convo_history = initiator.dialogue(responder, curr_time)

    return convo_history

def initialize(args):
    folder_name = args.output_folder_name  
    os.makedirs(folder_name, exist_ok=True)
    map_filename = args.map_filename
    agent_filenames = args.agent_filenames
    condition = args.condition
    start_date = args.start_date
    end_date = args.end_date
    default_agent_config_filename = args.default_agent_config_filename
    llm_provider = args.llm_provider
    llm_model_name = args.llm_model_name
    embedding_model_name = args.embedding_model_name
    daily_events_filename = args.daily_events_filename

    ## location
    generative_location = Location.from_yaml(map_filename)

    ## agents
    agents = []

    default_agent_kwargs = load_json_file(default_agent_config_filename)

    for agent_filename in agent_filenames:
        agent_kwargs = load_json_file(agent_filename)
        #inplace dict update
        agent_kwargs.update(default_agent_kwargs)
        agent_kwargs = override_agent_kwargs_with_condition(agent_kwargs, condition)
        agent_kwargs["llm_provider"] = llm_provider
        agent_kwargs["llm_model_name"] = llm_model_name
        agent_kwargs["embedding_model_name"] = embedding_model_name
        agent = HumanoidAgent(**agent_kwargs)
        agents.append(agent)

    ## time
    dates_of_interest = DatetimeNL.get_date_range(start_date, end_date)
    specific_times_of_interest = []
    for hour in range(6, 24):
        for minutes in ['00', '15', '30', '45']:
            hour_str = str(hour) if hour > 9 else '0' + str(hour)
            total_time = f"{hour_str}:{minutes}"
            specific_times_of_interest.append(total_time)
    
    ## daily_events
    curr_time_to_daily_event = get_curr_time_to_daily_event(daily_events_filename)

    return agents, dates_of_interest, specific_times_of_interest, generative_location, folder_name, curr_time_to_daily_event 

def parse_request(request, expected_keys=['curr_date', 'specific_time']):

    key_to_format = {
        'curr_date': "yyyy-mm-dd",
        'specific_time': "hh:mm",
        'name': "FirstName LastName",
        'conversation_history': 'list of { "name": name_self, "text": speak_self, "reaction": response_self}'
    }

    key_to_example = {
        'curr_date': "2023-01-03", 
        'specific_time': "09:00",
        'name': "John Lin",
        'conversation_history': '[{ "name": "John Lin", "text": "Hi there", "reaction": "Welcome Eddy back"}]'
    }

    key_to_options = {
        'name': list(name_to_agent.keys())
    }
    # automatically detect if data in GET or POST request form
    app_data = request.args if request.args else request.json

    # parameter validation
    useful_data = {}

    for key in expected_keys:
        if 'name' in key:
            short_key = 'name'
        else:
            short_key = key

        format = key_to_format[short_key]
        example = key_to_example[short_key]

        if key not in app_data:
            return f"{key} with required format {format} (e.g {example}) not in request"

        if short_key in key_to_options and app_data[key] not in key_to_options[short_key]:
            return f"For {key}, choose only from this list {key_to_options[short_key]}"
        useful_data[key] = app_data[key]
    
    return useful_data

@app.route('/plan_single', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def plan_single():

    data = parse_request(request, expected_keys=['curr_date', 'name'])
    
    # this is an error message
    if isinstance(data, str):
        return data
    curr_date = data['curr_date']
    name = data['name']

    curr_time = datetime.fromisoformat(curr_date)
    condition = curr_time_to_daily_event[curr_time] if curr_time in curr_time_to_daily_event else None
    plan = name_to_agent[name].plan(curr_time=curr_time, condition=condition)
    return plan 

@app.route('/plan', methods=['POST', 'GET'])
def plan():
    data = parse_request(request, expected_keys=['curr_date'])
    
    # this is an error message
    if isinstance(data, str):
        return data
    curr_date = data['curr_date']

    plans = []
    for _, agent in name_to_agent.items():
        curr_time = datetime.fromisoformat(curr_date)
        condition = curr_time_to_daily_event[curr_time] if curr_time in curr_time_to_daily_event else None
        plan = agent.plan(curr_time=curr_time, condition=condition)
        plans.append(plan)
    return plans

@app.route('/activity_single', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def get_15m_activity_single():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time', 'name'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    name = data['name']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

    logging.info(curr_date + ' ' + specific_time)

    overall_status = name_to_agent[name].get_status_json(curr_time, generative_location)
    logging.info("Overall status:")
    logging.info(json.dumps(overall_status, indent=4))
    return overall_status

@app.route('/activity', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def get_15m_activity():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')
    list_of_agent_statuses = []
    logging.info(curr_date + ' ' + specific_time)
    for _, agent in name_to_agent.items():
        # curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

        logging.info(curr_date + ' ' + specific_time)

        overall_status = agent.get_status_json(curr_time, generative_location)
        
        # agent.get_status_json(curr_time, generative_location)
        list_of_agent_statuses.append(overall_status)

        logging.info("Overall status:")
        logging.info(json.dumps(overall_status, indent=4))
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     futures = [executor.submit(agent.get_status_json, curr_time, generative_location) for _, agent in name_to_agent.items()]
    #     for future in as_completed(futures):
    #         overall_status = future.result()
    #         list_of_agent_statuses.append(overall_status)
    #         logging.info("Overall status:")
    #         logging.info(json.dumps(overall_status, indent=4))
    return list_of_agent_statuses

@app.route('/conversations', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def get_15m_conversations():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    list_of_agent_statuses = requests.get(
        url=urljoin(request.base_url, url_for("get_15m_activity")), 
        params = {
            "curr_date": curr_date,
            "specific_time": specific_time
    }).json()

    global agents
    location_to_agents = bucket_agents_by_location(list_of_agent_statuses, agents)
    # only 1 conversation per location, when there are 2 or more agents
    location_to_conversations = defaultdict(list)
    for location, agents in location_to_agents.items():
        if len(agents) > 1:
            selected_agents = random.sample(agents, 2)
            initiator, responder = selected_agents
            convo_history = requests.get(
                url=urljoin(request.base_url, url_for("chat")), 
                params = {
                    "curr_date": curr_date,
                    "specific_time": specific_time,
                    "initiator_name": initiator.name,
                    "responder_name": responder.name
            }).json()
            logging.info(f"Conversations at {location}")
            logging.info(json.dumps(convo_history, indent=4))
            location_to_conversations['-'.join(location)].append(convo_history)
    return location_to_conversations

@app.route('/logs', methods=['POST', 'GET'])
def write_logs():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')
    
    state_path = f"../generations/kuro_ai_universe/state_{curr_date}_{specific_time.replace(':', 'h')}.json"
    if os.path.exists(state_path):
        saved_state_data = json.load(open(state_path))
        return saved_state_data
    
    list_of_agent_statuses = []
    logging.info(curr_date + ' ' + specific_time)
    for _, agent in name_to_agent.items():
        # curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

        logging.info(curr_date + ' ' + specific_time)

        overall_status = agent.get_status_json(curr_time, generative_location)
        
        # agent.get_status_json(curr_time, generative_location)
        list_of_agent_statuses.append(overall_status)

        logging.info("Overall status:")
        logging.info(json.dumps(overall_status, indent=4))

    location_to_agents = bucket_agents_by_location(list_of_agent_statuses, [agent for _, agent in name_to_agent.items()])
    # only 1 conversation per location, when there are 2 or more agents
    location_to_conversations = defaultdict(list)
    for location, agents in location_to_agents.items():
        if len(agents) == 1:
            convo_history = agents[0].dialogue(agents[0], curr_time)
            logging.info(f"{agents[0].name}'s Thoughts at {location}")
            logging.info(json.dumps(convo_history, indent=4))
            location_to_conversations['-'.join(location)].append(convo_history)
        if len(agents) > 1:
            selected_agents = random.sample(agents, 2)
            initiator, responder = selected_agents
            convo_history = initiator.dialogue(responder, curr_time)
            logging.info(f"Conversations at {location}")
            logging.info(json.dumps(convo_history, indent=4))
            location_to_conversations['-'.join(location)].append(convo_history)

    overall_log = {
        "date": curr_date,
        "time": specific_time,
        "agents": list_of_agent_statuses,
        "conversations": {location: conversations for location, conversations in location_to_conversations.items()},
        "world": generative_location.to_json()
    }
    output_filename = f"{folder_name}/state_{curr_date}_{specific_time.replace(':','h')}.json"
    write_json_file(overall_log, output_filename)
    return overall_log

@app.route("/analytics", methods=['POST', 'GET'])
def get_analytics():
    data = request.json
    
    start_time_str = data.get('start_time')  # e.g., "2025-02-03 08:00"
    end_time_str = data.get('end_time')      # e.g., "2025-02-06 15:30"
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use 'YYYY-MM-DD HH:MM'."}), 400

    # Generate a list of time series with 30-minute intervals
    time_series = []
    if start_time.minute < 30:
        # Set minute to 0
        start_time = start_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Set minute to 30
        start_time = start_time.replace(minute=30, second=0, microsecond=0)
    current_time = start_time
    while current_time <= end_time:
        if 6 <= current_time.hour:  # Include only times from 06:00 onward
            time_series.append(current_time.strftime('%Y-%m-%d %H:%M'))
        current_time += timedelta(minutes=60)
    
    social_rel_time_series = []
    current_time = datetime(2025, 2, 4, 6, 0, 0)
    while current_time <= end_time:
        if 6 <= current_time.hour:  # Include only times from 06:00 onward
            social_rel_time_series.append(current_time.strftime('%Y-%m-%d %H:%M'))
        current_time += timedelta(minutes=60)
    
    timestamps = []
    analytics_data = dict()
    for name, _ in name_to_agent.items():
        analytics_data[name] = {
            "basicNeeds": {
                "timestamps": [],
                "energy": [],
                "health": [],
            },
            "socialRelationships": {},
            "activities": {
                "timestamps": [],
                "activities": [],
            },
            "emotions": {},
        }
    for time in time_series:
        data_file_name = f"state_{time.split(' ')[0]}_{time.split(' ')[1].replace(':', 'h')}.json"
        state_data = json.load(open(f"../generations/kuro_ai_universe/{data_file_name}"))
        timestamps.append(time)
        for agent in state_data["agents"]:
            analytics_data[agent['name']]['basicNeeds']['timestamps'].append(time)
            analytics_data[agent['name']]['basicNeeds']['energy'].append(agent['basic_needs']['energy'] * 10)
            analytics_data[agent['name']]['basicNeeds']['health'].append(agent['basic_needs']['health'] * 10)
            analytics_data[agent['name']]['emotions'][agent['emotion']] = analytics_data[agent['name']]['emotions'].get(agent['emotion'], 0) + 1
    
    for time in social_rel_time_series:
        data_file_name = f"state_{time.split(' ')[0]}_{time.split(' ')[1].replace(':', 'h')}.json"
        state_data = json.load(open(f"../generations/kuro_ai_universe/{data_file_name}"))
        for location, conversations_in_location in state_data['conversations'].items():
            for conversations in conversations_in_location:
                if len(conversations) < 2:
                    continue
                analytics_data[conversations[0]['name']]['socialRelationships'][conversations[1]['name']] = analytics_data[conversations[0]['name']]['socialRelationships'].get(conversations[1]['name'], 0) + len(conversations)

    last_state_data = json.load(open(f"../generations/kuro_ai_universe/state_{time_series[-1].split(' ')[0]}_{time_series[-1].split(' ')[1].replace(':', 'h')}.json"))
    for agent, analytics in analytics_data.items():
        relationship_data = []
        agent_social_relationships = {}
        for agent_state in last_state_data['agents']:
            if agent_state['name'] == agent:
                agent_social_relationships = agent_state['social_relationships']
                break
        
        for name, interaction in analytics['socialRelationships'].items():
            relationship_data.append({ "agent": name, "closeness": round(agent_social_relationships[name]['closeness'] / 15, 2) * 100, "interactions": interaction })
        analytics_data[agent]['socialRelationships'] = relationship_data
        
        agent_emotions = []
        total_emotion = 0
        for emotion, value in analytics['emotions'].items():
            total_emotion += value
        for emotion, value in analytics['emotions'].items():
            agent_emotions.append({ "emotion": emotion.title(), "percentage": round(value / total_emotion, 2) * 100 })
        analytics_data[agent]['emotions'] = agent_emotions
        
        current_activity = ""
        activity_count = 0
        for recent_time in time_series[::-1]:
            current_state = json.load(open(f"../generations/kuro_ai_universe/state_{recent_time.split(' ')[0]}_{recent_time.split(' ')[1].replace(':', 'h')}.json"))
            activity_data = ""
            location = ""
            for agent_state in current_state['agents']:
                if agent_state['name'] == agent:
                    activity_data = agent_state['activity']
                    location = agent_state['location'][0]
                    break
            if activity_data.split('>')[0].strip() == current_activity:
                continue
            current_activity = activity_data.split('>')[0].strip()
            analytics_data[agent]['activities']['timestamps'].append(recent_time)
            category_data = name_to_agent[agent].get_agent_activity_category(activity_data)
            category_data["timestamp"] = recent_time
            category_data["description"] = current_activity.title()
            category_data["location"] = location
            analytics_data[agent]['activities']['activities'].append(category_data)
            
            activity_count += 1
            if activity_count == 4:
                break
        
        analytics_data[agent]['activities']['activities'].reverse()
        analytics_data[agent]['activities']['timestamps'].reverse()
        
    return jsonify(analytics_data), 200

@app.route("/activity_log", methods=['POST'])
def get_activity_log():
    data = request.json
    
    start_time_str = data.get('start_time')  # e.g., "2025-02-03 08:00"
    end_time_str = data.get('end_time')      # e.g., "2025-02-06 15:30"
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use 'YYYY-MM-DD HH:MM'."}), 400

    # Generate a list of time series with 30-minute intervals
    time_series = []
    if start_time.minute < 30:
        # Set minute to 0
        start_time = start_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Set minute to 30
        start_time = start_time.replace(minute=30, second=0, microsecond=0)
    current_time = start_time
    while current_time <= end_time:
        if 6 <= current_time.hour:  # Include only times from 06:00 onward
            time_series.append(current_time.strftime('%Y-%m-%d %H:%M'))
        current_time += timedelta(minutes=60)
    
    activity_log = {
        "timestamp": "",
        "description": "",
        
    }

if __name__ == '__main__':
    logging.basicConfig(format='---%(asctime)s %(levelname)s \n%(message)s ---', level=logging.INFO)

    parser = argparse.ArgumentParser(description='run humanoid agents simulation')
    parser.add_argument("-o", "--output_folder_name", required=True)
    parser.add_argument("-m", "--map_filename", required=True) # '../locations/lin_family_map.yaml'
    parser.add_argument("-a", "--agent_filenames", required=True, nargs='+') # "../specific_agents/john_lin.json", "../specific_agents/eddy_lin.json"
    parser.add_argument("-da", "--default_agent_config_filename", default="default_agent_config.json")
    parser.add_argument('-s', '--start_date', help='Enter start date (inclusive) by YYYY-MM-DD e.g.2023-01-03', default="2023-01-03")
    parser.add_argument('-e', '--end_date', help='Enter end date (inclusive) by YYYY-MM-DD e.g.2023-01-04', default="2023-01-03")
    parser.add_argument("-c", "--condition", default=None, choices=["disgusted", "afraid", "sad", "surprised", "happy", "angry", "neutral", 
                                            "fullness", "social", "fun", "health", "energy", 
                                            "closeness_0", "closeness_5", "closeness_10", "closeness_15", None])
    parser.add_argument("-l", "--llm_provider", default="openai", choices=["openai", "local", "mindsdb"])
    parser.add_argument("-lmn", "--llm_model_name", default="gpt-3.5-turbo")
    parser.add_argument("-emn", "--embedding_model_name", default="text-embedding-ada-002", help="with local, please use all-MiniLM-L6-v2 or another name compatible with SentenceTransformers")
    parser.add_argument("-daf", "--daily_events_filename", default=None)


    args = parser.parse_args()
    logging.info(args)
    agents, dates_of_interest, specific_times_of_interest, generative_location, folder_name, curr_time_to_daily_event = initialize(args)
    name_to_agent = {agent.name: agent for agent in agents}
    print("starting")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
