from tqdm import tqdm
from collections import Counter
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import json
import pandas as pd
import numpy as np
import warnings
import random
import time
import multiprocessing
import threading
from multiprocessing import Process, Queue, current_process
import scipy as sp
from functools import partial
from numba import jit

warnings.filterwarnings('ignore')

mutex = threading.Lock()

def read_data(use_reduced):
    start = time.time()
    print("Start read data")

    print('Reading train.csv file....')
    if use_reduced:
        train = pd.read_csv('../input/data-science-bowl-2019/train.csv', nrows=450000)
    else:
        train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    if use_reduced:
        test = pd.read_csv('../input/data-science-bowl-2019/test.csv', nrows=3000)
    else:
        test = pd.read_csv('../input/data-science-bowl-2019/test.csv')


    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    print("read data done, time - ", time.time() - start)
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    start = time.time()
    print("Start encoding data")

    games = list(set(train[train['type'] == 'Game']['title']).union(
        set(test[test['type'] == 'Game']['title'])))

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    train['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['world']))
    test['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['world']))
    all_type_world = list(set(train["type_world"].unique()).union(test["type_world"].unique()))

    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
        set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    print("End encoding data, time - ", time.time() - start)

    assessments = {
        'Chest Sorter (Assessment)': 0,
        'Bird Measurer (Assessment)': 1,
        'Mushroom Sorter (Assessment)': 2,
        'Cart Balancer (Assessment)': 3,
        'Cauldron Filler (Assessment)': 4,
    }

    event_data = {}
    event_data["train_labels"] = train_labels
    event_data["win_code"] = win_code
    event_data["list_of_user_activities"] = list_of_user_activities
    event_data["list_of_event_code"] = list_of_event_code
    event_data["activities_labels"] = activities_labels
    event_data["assess_titles"] = assess_titles
    event_data["list_of_event_id"] = list_of_event_id
    event_data["all_title_event_code"] = all_title_event_code
    event_data["activities_map"] = activities_map
    event_data["all_type_world"] = all_type_world
    event_data["description"] = 0
    event_data['games'] = games
    event_data['assessments'] = assessments

    game_difficult = {
        'Chest Sorter (Assessment)': 122,
        'Bird Measurer (Assessment)': 113,
        'Mushroom Sorter (Assessment)': 89,
        'Cart Balancer (Assessment)': 87,
        'Cauldron Filler (Assessment)': 85,
    }
    event_data['game_difficult'] = game_difficult
    event_data['sa'] = 12
    event_data['divider'] = 40
    event_data['pow'] = 35

    return train, test, event_data

def get_all_features(feature_dict, ac_data):
    if len(ac_data['durations']) > 0:
        feature_dict['installation_duration_mean'] = np.mean(ac_data['durations'])
        feature_dict['installation_duration_sum'] = np.sum(ac_data['durations'])
    else:
        feature_dict['installation_duration_mean'] = 0
        feature_dict['installation_duration_sum'] = 0

    return feature_dict

def cnt_miss(df):
    misses = []
    levels = []
    rounds = []
    duration = []
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        js_obj = json.loads(x)
        y = 0
        if 'misses' in js_obj:
            y = js_obj['misses']

        if 'round' in js_obj:
            rounds.append(js_obj['round'])
        if 'level' in js_obj:
            levels.append(int(json.loads(x)['level']))

        duration.append(int(js_obj['duration']))
        misses.append(y)
    return misses, rounds, levels, duration

def get_data(user_sample, event_data, test_set):
    all_assessments = []

    current_example = {
        'installation_id': user_sample['installation_id'].iloc[0],
        "current_assessment": 0,

        'sessions_count': 0,
        "assessment_session_count": 0,
        'clip_session_count': 0,
        'game_session_count': 0,
        'activity_session_count': 0,

        "all_event_count": 0,
        "assessment_event_count": 0,
        'clip_event_count': 0,
        'game_event_count': 0,
        'activity_event_count': 0,

        'last_durations': 0,
        "durations_mean": 0,

        "assessment_durations": 0,
        'assessment_last_durations': 0,
        "assessment_durations_mean": 0,

        "clip_durations": 0,
        "clip_durations_mean": 0,

        "game_durations": 0,
        'game_last_durations': 0,
        "game_durations_mean": 0,

        "activity_durations": 0,
        'activity_last_durations': 0,
        "activity_durations_mean": 0,

        'accumulated_correct_attempts': 0,
        'accumulated_uncorrect_attempts': 0,
        'accumulated_accuracy': 0,
        'accumulated_accuracy_group': 0,
        'accuracy_group': 0,

        'try_solve_assessment': 0,
        'not_try_solve_assessment': 0,
        'try_solve_proc': 0,
    }

    for i in event_data["list_of_user_activities"]:
        current_example[i + "_correct"] = 0
        current_example[i + "_uncorrect"] = 0
        current_example[i + "_correct_proc"] = 0
        current_example[i + "_uncorrect_proc"] = 0

    all_event_count = 0
    sessions_count = 0
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0

    try_solve_assessment = 0
    not_try_solve_assessment = 0

    durations = 0
    last_durations = 0

    true_attempts = 0
    false_attempts = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0

    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    event_code_count = {ev: 0 for ev in event_data["list_of_event_code"]}
    event_code_proc_count = {str(ev) + "_proc" : 0. for ev in event_data["list_of_event_code"]}

    event_id_count = {eve: 0 for eve in event_data["list_of_event_id"]}
    event_id_proc_count = {eve + "_proc": 0 for eve in event_data["list_of_event_id"]}

    title_event_code_count = {t_eve: 0 for t_eve in event_data["all_title_event_code"]}
    title_count = {eve: 0 for eve in event_data["activities_labels"].values()}

    title_correct = {eve + "_correct": 0 for eve in event_data["activities_labels"].values()}
    title_uncorrect = {eve + "_uncorrect": 0 for eve in event_data["activities_labels"].values()}
    title_correct_proc = {eve + "_correct_proc": 0 for eve in event_data["activities_labels"].values()}

    type_world_count = {w_eve: 0 for w_eve in event_data["all_type_world"]}

    last_accuracy_title = {'acc_' + title: -1 for title in event_data["assess_titles"]}
    last_game_time_title = {'lgt_' + title: 0 for title in event_data["assess_titles"]}
    ac_game_time_title = {'agt_' + title: 0 for title in event_data["assess_titles"]}
    ac_true_attempts_title = {'ata_' + title: 0 for title in event_data["assess_titles"]}
    ac_false_attempts_title = {'afa_' + title: 0 for title in event_data["assess_titles"]}

    assessment_session_count = 0
    assessment_event_count = 0
    assessment_durations = 0
    assessment_last_durations = 0

    clip_session_count = 0
    clip_event_count = 0
    clip_durations = 0
    clip_last_durations = 0

    game_session_count = 0
    game_event_count = 0
    game_durations = 0
    game_last_durations = 0

    activity_session_count = 0
    activity_event_count = 0
    activity_durations = 0
    activity_last_durations = 0

    accumulated_game_miss = 0
    Cauldron_Filler_4025 = 0
    mean_game_round = 0
    mean_game_duration = 0
    mean_game_level = 0
    chest_assessment_uncorrect_sum = 0

    elo_like_rating = 100
    elo_like_rating_cart_balancer = 100

    # GAMES
    chow_Time_overweight = 0
    chow_Time_underweight = 0

    games_count = {title + "_count": 0 for title in event_data["list_of_user_activities"]}
    games_misses = {title + "_misses": 0 for title in event_data["list_of_user_activities"]}
    games__misses_mean_summ = {title + "_misses_mean_summ": 0 for title in event_data["list_of_user_activities"]}
    games_average_max_level = {title + "_average_max_level": 0 for title in event_data["list_of_user_activities"]}
    games_misses_mean_mean = {title + "_misses_mean_mean": 0 for title in event_data["list_of_user_activities"]}
    games_misses_mean_overall = {title + "_misses_mean_overall": 0 for title in event_data["list_of_user_activities"]}
    games_rounds_sum = {title + "_rounds_sum": 0 for title in event_data["list_of_user_activities"]}
    games_rounds_mean = {title + "_rounds_mean": 0 for title in event_data["list_of_user_activities"]}


    game_time_dict = {'Clip_gametime': 0, 'Game_gametime': 0,
                      'Activity_gametime': 0, 'Assessment_gametime': 0}

    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy': 0,
                            'Mushroom Sorter (Assessment)_4020_accuracy': 0,
                            'Bird Measurer (Assessment)_4020_accuracy': 0,
                            'Chest Sorter (Assessment)_4020_accuracy': 0}

    counter = 0

    for i, session in user_sample.groupby('game_session', sort=False):
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = event_data["activities_labels"][session_title]

        session_event_count = session["event_code"].count()

        session_duration = (session.iloc[-1, 2] - session.iloc[0, 2]).seconds

        true_events = session['event_data'].str.contains('true').sum()
        false_events = session['event_data'].str.contains('false').sum()

        if session_title_text in event_data['assessments']:
            current_example['current_assessment'] = event_data['assessments'][session_title_text]
        else:
            event_data['assessments'][session_title_text] = len(event_data['assessments'])
            current_example['current_assessment'] = event_data['assessments'][session_title_text]

        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            all_attempts = session.query(f'event_code == {event_data["win_code"][session_title]}')

            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            true_events -= true_attempts
            false_events -= false_attempts

            current_example.update(accuracy_groups)
            current_example.update(event_code_count.copy())

            current_example.update(event_code_proc_count.copy())
            current_example.update(event_id_count.copy())
            current_example.update(event_id_proc_count.copy())

            current_example.update(title_event_code_count.copy())
            current_example.update(title_count.copy())
            current_example.update(type_world_count.copy())
            current_example.update(game_time_dict.copy())

            current_example.update(last_accuracy_title.copy())
            current_example.update(ac_true_attempts_title.copy())
            current_example.update(ac_false_attempts_title.copy())
            current_example.update(last_game_time_title.copy())
            current_example.update(ac_game_time_title.copy())

            current_example.update(assess_4020_acc_dict.copy())

            current_example["all_event_count"] = all_event_count

            current_example['try_solve_assessment'] = try_solve_assessment
            current_example['not_try_solve_assessment'] = not_try_solve_assessment

            divider = current_example['try_solve_assessment']
            current_example['accumulated_accuracy_group'] = accumulated_accuracy_group  / divider if divider > 0 else 0

            divider = current_example['try_solve_assessment'] + current_example['not_try_solve_assessment']
            current_example['try_solve_proc'] = current_example['try_solve_assessment'] / divider if divider > 0 else 0

            current_example["sessions_count"] = sessions_count
            current_example["durations"] = durations
            current_example["last_durations"] = last_durations

            current_example["durations_mean"] = durations / (sessions_count + 1)

            current_example["accumulated_correct_attempts"] = accumulated_correct_attempts
            current_example["accumulated_uncorrect_attempts"] = accumulated_uncorrect_attempts

            current_example['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0

            current_example['assessment_session_count'] += assessment_session_count
            current_example['assessment_event_count'] = assessment_event_count
            current_example['assessment_durations'] = assessment_durations
            current_example['assessment_last_durations'] = assessment_last_durations
            current_example['assessment_durations_mean'] = assessment_durations / assessment_session_count if assessment_session_count > 0 else 0

            current_example['clip_session_count'] += clip_session_count
            current_example['clip_event_count'] = clip_event_count
            current_example['clip_durations'] = clip_durations
            current_example['clip_last_durations'] = clip_last_durations
            current_example['clip_durations_mean'] = clip_durations / clip_session_count if clip_session_count > 0 else 0

            current_example['game_session_count'] += game_session_count
            current_example['game_event_count'] = game_event_count
            current_example['game_durations'] = game_durations
            current_example['game_last_durations'] = game_last_durations
            current_example['game_durations_mean'] = game_durations / game_session_count if game_session_count > 0 else 0

            current_example['activity_session_count'] += activity_session_count
            current_example['activity_event_count'] = activity_event_count
            current_example['activity_durations'] = activity_durations
            current_example['activity_last_durations'] = activity_last_durations
            current_example['activity_durations_mean'] = activity_durations / activity_session_count if activity_session_count > 0 else 0

            current_example['accumulated_game_miss'] = accumulated_game_miss
            current_example['Cauldron_Filler_4025'] = Cauldron_Filler_4025 / counter if counter > 0 else 0
            current_example['mean_game_round'] = mean_game_round
            current_example['mean_game_duration'] = mean_game_duration
            current_example['mean_game_level'] = mean_game_level
            current_example['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum

            current_example['elo_like_rating'] = elo_like_rating
            current_example['elo_like_rating_cart_balancer'] = elo_like_rating_cart_balancer

            current_example['Chow_Time_overweight'] = chow_Time_overweight
            current_example['Chow_Time_underweight'] = chow_Time_underweight
            if 'Chow_Time' in event_data["games"]:
                current_example['chow_time_overweight_mean'] = current_example['Chow_Time_overweight'] / current_example['Chow_Time_count']
                current_example['chow_time_underweight_mean'] = current_example['Chow_Time_underweight'] / current_example['Chow_Time_count']

            current_example.update(games_count.copy())
            current_example.update(games_misses.copy())
            current_example.update(games__misses_mean_summ.copy())
            current_example.update(games_average_max_level.copy())
            current_example.update(games_misses_mean_mean.copy())
            current_example.update(games_misses_mean_overall.copy())
            current_example.update(games_rounds_sum.copy())
            current_example.update(games_rounds_mean.copy())

            current_example.update(title_correct.copy())
            current_example.update(title_uncorrect.copy())
            current_example.update(title_correct_proc.copy())

            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            last_accuracy_title['acc_' + session_title_text] = accuracy
            accumulated_accuracy += accuracy

            current_example['accuracy_group'] = accuracy

            if accuracy == 0:
                accuracy_group = 0
            elif accuracy == 1:
                accuracy_group = 3
            elif accuracy == 0.5:
                accuracy_group = 2
            else:
                accuracy_group = 1

            if test_set:
                last_assesment = current_example.copy()

            if true_attempts + false_attempts > 0:
                all_assessments.append(current_example.copy())
                accuracy_groups[accuracy_group] += 1
                try_solve_assessment += 1
            if true_attempts + false_attempts == 0 and len(session) > 1:
                not_try_solve_assessment += 1

            accumulated_accuracy_group += current_example['accuracy_group']
            counter += 1

        if session_type == 'Assessment':
            assessment_session_count += 1
            assessment_event_count += session_event_count
            assessment_durations += session_duration
            assessment_last_durations = session_duration

            game_s = session[session.event_code == 2030]
            misses, rounds, levels, duration = cnt_miss(game_s)

            misses_cnt = sum(misses)
            accumulated_game_miss += misses_cnt

            games_count[session_title_text + "_count"] += 1
            games_misses[session_title_text + "_misses"] += misses_cnt

            if len(misses) > 0:
                games__misses_mean_summ[session_title_text + "_misses_mean_summ"] += misses_cnt / len((misses))

            if len(levels) > 0:
                games_average_max_level[session_title_text + "_average_max_level"] = \
                    (games_average_max_level[session_title_text + "_average_max_level"] + max(levels)) / 2

            games_rounds_sum[session_title_text + "_rounds_sum"] += len(rounds)
            games_rounds_mean[session_title_text + "_rounds_sum"] = \
                games_rounds_sum[session_title_text + "_rounds_sum"] / games_count[session_title_text + "_count"]

            games_misses_mean_mean[session_title_text + "_misses_mean_mean"] = \
                games__misses_mean_summ[session_title_text + "_misses_mean_summ"] / games_count[
                    session_title_text + "_count"]
            games_misses_mean_overall[session_title_text + "_misses_mean_overall"] = games_misses[
                                                                                         session_title_text + "_misses"] / \
                                                                                     games_count[
                                                                                         session_title_text + "_count"]

            if session_title_text in event_data['game_difficult']:
                diff = event_data['game_difficult'][session_title_text]
            else:
                diff = 100

            for i in range(false_attempts):
                ea = 1 / (1 + event_data['pow'] ** ((diff - elo_like_rating) / event_data['divider']))
                elo_like_rating += event_data['sa'] * (-ea)
            for i in range(true_attempts):
                ea = 1 / (1 + event_data['pow'] ** ((diff - elo_like_rating) / event_data['divider']))
                elo_like_rating += event_data['sa'] * (1 - ea)

            if session_title_text == 'Cart Balancer (Assessment)':
                elo_like_rating_cart_balancer = elo_for_cart_balancer_assessment(session, elo_like_rating_cart_balancer, event_data)

            ac_true_attempts_title['ata_' + session_title_text] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text] += false_attempts
            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]

        elif session_type == 'Clip':
            clip_session_count += 1
            clip_event_count += session_event_count
            clip_durations += session_duration
            clip_last_durations = session_duration

        elif session_type == 'Game':
            game_session_count += 1
            game_event_count += session_event_count
            game_durations += session_duration
            game_last_durations = session_duration

            game_s = session[session.event_code == 2030]
            misses, rounds, levels, duration = cnt_miss(game_s)

            misses_cnt = sum(misses)
            accumulated_game_miss += misses_cnt

            games_count[session_title_text + "_count"] += 1
            games_misses[session_title_text + "_misses"] += misses_cnt

            if len(misses) > 0:
                games__misses_mean_summ[session_title_text + "_misses_mean_summ"] += misses_cnt / len((misses))

            if len(levels) > 0:
                games_average_max_level[session_title_text + "_average_max_level"] = \
                    (games_average_max_level[session_title_text + "_average_max_level"] + max(levels)) / 2

            games_rounds_sum[session_title_text + "_rounds_sum"] += len(rounds)
            games_rounds_mean[session_title_text + "_rounds_sum"] = \
                games_rounds_sum[session_title_text + "_rounds_sum"] / games_count[session_title_text + "_count"]

            games_misses_mean_mean[session_title_text + "_misses_mean_mean"] = \
                games__misses_mean_summ[session_title_text + "_misses_mean_summ"] / games_count[session_title_text + "_count"]
            games_misses_mean_overall[session_title_text + "_misses_mean_overall"] = games_misses[session_title_text + "_misses"] / games_count[session_title_text + "_count"]

            if session_title_text == 'Chow Time':
                overweight = session['event_data'].str.contains("That's too much food").sum()
                underweight = session['event_data'].str.contains("That's not enough food").sum()
                chow_Time_overweight += overweight
                chow_Time_underweight += underweight

            try:
                game_round = json.loads(session['event_data'].iloc[-1])["round"]
                mean_game_round = (mean_game_round + game_round) / 2.0
            except:
                pass

            try:
                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]
                mean_game_duration = (mean_game_duration + game_duration) / 2.0
            except:
                pass

            try:
                game_level = json.loads(session['event_data'].iloc[-1])["level"]
                mean_game_level = (mean_game_level + game_level) / 2.0
            except:
                pass

        elif session_type == 'Activity':
            activity_session_count += 1
            activity_event_count += session_event_count
            activity_durations += session_duration
            activity_last_durations = session_duration

        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                if col == 'title':
                    x = event_data["activities_labels"][k]
                elif col == 'world':
                    x = event_data["all_type_world"][k]
                else:
                    x = k
                counter[x] += num_of_session_count[k]
            return counter

        def update_proc(count: dict):
            res = {}
            for k, val in count.items():
                res[str(k) + "_proc"] = (float(val)) / session_event_count
            return res

        event_code_count = update_counters(event_code_count, "event_code")
        event_code_proc_count = update_proc(event_code_count)

        event_id_count = update_counters(event_id_count, "event_id")
        event_id_proc_count = update_proc(event_id_count)

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')
        title_count = update_counters(title_count, 'title')

        type_world_count = update_counters(type_world_count, 'world')

        assess_4020_acc_dict = get_4020_acc(session, assess_4020_acc_dict, event_data)
        game_time_dict[session_type + '_gametime'] = (game_time_dict[session_type + '_gametime'] + (
                    session['game_time'].iloc[-1] / 1000.0)) / 2.0

        all_event_count += session_event_count
        sessions_count +=1

        durations += session_duration
        last_durations = session_duration

        accumulated_correct_attempts += true_attempts
        accumulated_uncorrect_attempts += false_attempts

        title_correct[session_title_text + "_correct"] += true_events
        title_uncorrect[session_title_text + "_uncorrect"] += false_events
        devider = title_correct[session_title_text + "_correct"] + title_uncorrect[session_title_text + "_uncorrect"]
        title_correct_proc[session_title_text + "_correct_proc"] / devider if devider > 0 else 0

        Assess_4025 = session[(session.event_code == 4025) & (session.title == 'Cauldron Filler (Assessment)')]
        true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
        false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()

        cau_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (true_attempts_ + false_attempts_) != 0 else 0
        Cauldron_Filler_4025 += cau_assess_accuracy_
        chest_assessment_uncorrect_sum += len(session[session.event_id == "df4fe8b6"])

    if test_set:
        return last_assesment, all_assessments
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_4020_acc(df, counter_dict, event_data):
    for e in ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)',
              'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)']:
        Assess_4020 = df[(df.event_code == 4020) & (df.title == event_data["activities_map"][e])]
        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()

        measure_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (
                                                                                                      true_attempts_ + false_attempts_) != 0 else 0
        counter_dict[e + "_4020_accuracy"] += (counter_dict[e + "_4020_accuracy"] + measure_assess_accuracy_) / 2.0

    return counter_dict

def get_users_data(users_list, return_dict,  event_data, test_set):
    if test_set:
        for user in users_list:
            return_dict.append(get_data(user, event_data, test_set))
    else:
        answer = []
        for user in users_list:
            answer += get_data(user, event_data, test_set)
        return_dict += answer

def get_data_parrallel(users_list, event_data, test_set):
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    threads_number = event_data["process_numbers"]
    data_len = len(users_list)
    processes = []
    cur_start = 0
    cur_stop = 0
    for index in range(threads_number):
        cur_stop += (data_len-1) // threads_number

        if index != (threads_number - 1):
            p = Process(target=get_users_data, args=(users_list[cur_start:cur_stop], return_dict, event_data, test_set))
        else:
            p = Process(target=get_users_data, args=(users_list[cur_start:], return_dict, event_data, test_set))

        processes.append(p)
        cur_start = cur_stop

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    return list(return_dict)


def get_train_and_test(train, test, event_data):
    start = time.time()
    print("Start get_train_and_test")

    compiled_train = []
    compiled_test = []

    user_train_list = []
    user_test_list = []

    stride_size = event_data["strides"]
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):
        user_train_list.append(user_sample)
        if (i + 1) % stride_size == 0:
            compiled_train += get_data_parrallel(user_train_list, event_data, False)
            del user_train_list
            user_train_list = []

    if len(user_train_list) > 0:
        compiled_train += get_data_parrallel(user_train_list, event_data, False)
        del user_train_list

    for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby('installation_id', sort=False)), total=1000):
        user_test_list.append(user_sample)
        if (i + 1) % stride_size == 0:
            compiled_test += get_data_parrallel(user_test_list, event_data, True)
            del user_test_list
            user_test_list = []

    if len(user_test_list) > 0:
        compiled_test += get_data_parrallel(user_test_list, event_data, True)
        del user_test_list

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = [x[0] for x in compiled_test]

    reduce_train_from_test = []
    for i in [x[1] for x in compiled_test]:
        reduce_train_from_test += i

    reduce_test = pd.DataFrame(reduce_test)
    reduce_train_from_test = pd.DataFrame(reduce_train_from_test)
    print("End get_train_and_test, time - ", time.time() - start)
    return reduce_train, reduce_test, reduce_train_from_test


def get_train_and_test_single_proc(train, test, event_data):
    compiled_train = []
    compiled_test = []
    compiled_test_his = []
    for ins_id, user_sample in tqdm(train.groupby('installation_id', sort=False), total=17000):
        compiled_train += get_data(user_sample, event_data, False)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
        test_data = get_data(user_sample, event_data, True)
        compiled_test.append(test_data[0])
        compiled_test_his += test_data[1]

    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    reduce_test_his = pd.DataFrame(compiled_test_his)

    return reduce_train, reduce_test, reduce_test_his


def predict(sample_submission, y_pred):
    sample_submission['accuracy_group'] = y_pred
    sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)
    sample_submission.to_csv('submission.csv', index=False)
    print(sample_submission['accuracy_group'].value_counts(normalize=True))


def get_random_assessment(reduce_train):
    used_idx = []
    for iid in tqdm(set(reduce_train['installation_id'])):
        list_ = list(reduce_train[reduce_train['installation_id']==iid].index)
        cur = random.choices(list_, k = 1)[0]
        used_idx.append(cur)
    reduce_train_t = reduce_train.loc[used_idx]
    return reduce_train_t, used_idx


def elo_for_cart_balancer_assessment(session, elo_like_rating, event_data):
    try:
        game_start_data = session.query(f'event_code == 2020')
        if len(game_start_data['event_data']) > 0:
            x = game_start_data['event_data'].iloc[0]
            js_obj = json.loads(x)

            left_sum = 0
            rigth_sum = 0

            if 'crystals' in js_obj:
                crystals = js_obj["crystals"]

            max_weigth = max([x['weight'] for x in crystals])
            game_step_data = session.query(f'event_code == 4020')
            if len(game_step_data['event_data']) > 0:
                for i in range(len(game_step_data['event_data'])):
                    js_obj = json.loads(game_step_data['event_data'].iloc[i])

                    diff1 = 70
                    koef1 = 0.013
                    if js_obj["weight"] == max_weigth:
                        ea = 1 / (1 + event_data['pow'] ** ((diff1 - elo_like_rating) / event_data['divider']))
                        elo_like_rating += (koef1 * event_data['sa'] * (1 - ea))
                    else:
                        ea = 1 / (1 + event_data['pow'] ** ((diff1 - elo_like_rating) / event_data['divider']))
                        elo_like_rating += (koef1 * event_data['sa'] * (-ea))

                    diff2 = 50
                    koef2 = 0.2

                    if 'side' in js_obj:
                        point = 0
                        if left_sum > rigth_sum and js_obj['side'] != 'left':
                            point = 1
                        if left_sum < rigth_sum and js_obj['side'] == 'left':
                            point = 1
                        if left_sum == rigth_sum:
                            point = 1

                        ea = 1 / (1 + event_data['pow'] ** ((diff2 - elo_like_rating) / event_data['divider']))
                        elo_like_rating += (koef1 * event_data['sa'] * (point-ea))


                    left_sum = sum([x['weight'] for x in js_obj['left']])
                    rigth_sum = sum([x['weight'] for x in js_obj['right']])

                    new_crystals = []
                    if 'crystals' in js_obj:
                        new_crystals = js_obj["crystals"]

                    if len(new_crystals) > 0:
                        max_weigth = max([x['weight'] for x in new_crystals])
                    else:
                        break
    except:
        print("hi")

    return elo_like_rating


def main():
    in_kaggle = False
    use_parallel = True
    use_reduced_dataset = False
    use_validate = True
    random.seed(42)
    start_program = time.time()

    event_data = {}
    if in_kaggle:
        event_data["strides"] = 250
        event_data["process_numbers"] = 4
    else:
        event_data["strides"] = 1500
        event_data["process_numbers"] = 4

    train, test, train_labels, specs, sample_submission = read_data(use_reduced_dataset)
    train, test, event_data_update = encode_title(train, test, train_labels)
    event_data.update(event_data_update)

    if use_parallel:
        reduce_train, reduce_test, reduce_train_from_test = get_train_and_test(train, test, event_data)
    else:
        reduce_train, reduce_test, reduce_train_from_test = get_train_and_test_single_proc(train, test, event_data)

    dels = [train, test]
    del dels

    reduce_train.sort_values("installation_id", axis=0, ascending=True, inplace=True, na_position='last')
    reduce_test.sort_values("installation_id", axis=0, ascending=True, inplace=True, na_position='last')
    reduce_train = pd.concat([reduce_train, reduce_train_from_test], ignore_index=True)
    reduce_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]
    reduce_test.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]

    reduce_train.fillna(0)
    reduce_test.fillna(0)

    features = list(reduce_train.columns)
    features.remove('accuracy_group')

    print(u"Features count - ", len(features))

    drop_columns = []
    for feature in features:
        if feature == 'installation_id':
            continue

        if np.sum(reduce_train[feature].values) < 1:
            print(feature)
            drop_columns.append(feature)

    reduce_train = reduce_train.drop(drop_columns, axis=1)
    reduce_test = reduce_test.drop(drop_columns, axis=1)

    reduce_train.to_csv('reduce_train.csv', index=False, sep=";")
    reduce_test.to_csv('reduce_test.csv', index=False, sep=";")

    reduce_train = pd.read_csv('reduce_train.csv', sep=";")
    reduce_test = pd.read_csv('reduce_test.csv', sep=";")

    features = list(reduce_train.columns)
    features.remove('accuracy_group')
    features.remove('installation_id')

    sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

    if 'installation_id' not in features:
        features.append('installation_id')
    print(features)

    reduce_train = reduce_train

    model = AnsambleModel(reduce_train, features, kmodels=6, kfold=6)
    train_predict = model.predict(reduce_train)

    oof_pred = trashold(train_predict)

    y_val1 = reduce_train['accuracy_group'].values.copy()
    y_val = y_val1.copy()
    y_val[y_val1 > 0.5] = 3
    y_val[y_val1 == 0.5] = 2
    y_val[y_val1 < 0.5] = 1
    y_val[y_val1 == 0.0] = 0

    print("Cappa", qwk(y_val, oof_pred))
    print("RMSE", np.sqrt(mean_squared_error(y_val, oof_pred)))

    optR = OptimizedRounder()
    optR.fit(train_predict, y_val)
    coefficients = optR.coefficients()
    print("New coefs = ", coefficients)
    opt_preds = optR.predict(train_predict, coefficients)
    print("New train cappa rounding= ", qwk(y_val, opt_preds))
    print("RMSE", np.sqrt(mean_squared_error(y_val, opt_preds)))

    y_final = model.predict(reduce_test)
    oof_pred = trashold(y_final)

    predict(sample_submission, oof_pred)
    print("Programm full time:", time.time() - start_program)

def trashold(y_final):
    bound = [0.33530578, 0.51589898, 0.71000847]
    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    oof_pred = np.array(list(map(classify, y_final)))

    return  oof_pred

def qwk2(a1, a2):
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

@jit
def qwk(a1, a2):
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

class AnsambleModel:
    def __init__(self, train, features, kmodels=5, kfold=3):
        self.bin_models = []
        self.models = []
        self.features = features
        self.kfold = kfold

        oof_rmse_scores_lgb = []
        oof_cohen_scores_lgb = []

        oof_rmse_scores_lgb2 = []
        oof_cohen_scores_lgb2 = []

        oof_rmse_scores = []
        oof_cohen_scores = []

        target = 'accuracy_group'

        for model_number in range(kmodels):
            print("\n\n\nTrain models banch #", model_number, "\n")
            kf = GroupKFold(n_splits=kfold)
            #kf = KFold(n_splits=kfold, shuffle=True)
            #kf = GroupShuffleSplit(n_splits=kfold, test_size=0.2, random_state=13)

            oof_pred1 = np.zeros(len(train))
            oof_pred2 = np.zeros(len(train))
            ind = []

            for fold, (tr_ind, val_ind) in enumerate(kf.split(train, groups=train['installation_id'])):

                print('Fold:', fold + 1)
                x_train, x_val = train[features].iloc[tr_ind], train[features].iloc[val_ind]
                y_train1, y_val1 = train[target][tr_ind], train[target][val_ind]

                y_train = y_train1.copy()
                y_train[y_train1 > 0.5] = 3
                y_train[y_train1 == 0.5] = 2
                y_train[y_train1 < 0.5] = 1
                y_train[y_train1 == 0.0] = 0

                y_val = y_val1.copy()
                y_val[y_val1 > 0.5] = 3
                y_val[y_val1 == 0.5] = 2
                y_val[y_val1 < 0.5] = 1
                y_val[y_val1 == 0.0] = 0

                x_train.drop('installation_id', inplace=True, axis=1)
                x_val, idx_val = get_random_assessment(x_val)

                ind.extend(idx_val)
                x_val.drop('installation_id', inplace=True, axis=1)
                y_val1 = y_val1.loc[idx_val]

                train_set = lgb.Dataset(x_train, y_train1, categorical_feature=['current_assessment'], free_raw_data=False)
                val_set = lgb.Dataset(x_val, y_val1, categorical_feature=['current_assessment'], free_raw_data=False)

                params = {
                    'num_boost_round': 1400,
                    'boosting_type': 'gbdt',
                    'metric': ['root_mean_squared_error', 'binary'],
                    'objective': 'regression',
                    'n_jobs': -1,
                    'seed': 42,
                    'num_leaves': 20,
                    'learning_rate': 0.01,
                    'max_depth': 10,
                    'lambda_l1': 2.0,
                    'lambda_l2': 1.0,
                    'bagging_fraction': 0.4,
                    'bagging_freq': 1,
                    'feature_fraction': 0.3,
                    'early_stopping_rounds': 250,
                    'first_metric_only': 'true',
                    'verbose': 0,
                }

                model1 = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)

                reg_pred1 = model1.predict(x_val)
                oof_pred1[idx_val] = reg_pred1
                print('Our fold cohen kappa score for lbg is:', qwk2(y_val.loc[idx_val].values, trashold(reg_pred1)))

                self.models.append(model1)

                params2 = {
                    'num_boost_round': 1400,
                    'boosting_type': 'gbdt',
                    'metric': ['root_mean_squared_error', 'binary'],
                    'objective': 'regression',
                    'n_jobs': -1,
                    'seed': 85,
                    'num_leaves': 15,
                    'learning_rate': 0.01,
                    'max_depth': 14,
                    'lambda_l1': 2.0,
                    'lambda_l2': 1.0,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'feature_fraction': 0.6,
                    'early_stopping_rounds': 400,
                    'first_metric_only': 'true',
                    'verbose': 0,
                }

                model2 = lgb.train(params2, train_set, valid_sets=[train_set, val_set], verbose_eval=100)

                reg_pred2 = model2.predict(x_val)
                oof_pred2[idx_val] = reg_pred2
                print('Our fold cohen kappa score for lbg2 is:', qwk2(y_val.loc[idx_val].values, trashold(reg_pred2)))

                self.models.append(model2)

            train_convert = train[target][ind].copy()
            train_convert[train[target][ind] > 0.5] = 3
            train_convert[train[target][ind] == 0.5] = 2
            train_convert[train[target][ind] < 0.5] = 1
            train_convert[train[target][ind] == 0.0] = 0

            oof_rmse_score_lgb = np.sqrt(mean_squared_error(train_convert.values, trashold(oof_pred1[ind])))
            oof_cohen_score_lgb = qwk(train_convert.values, trashold(oof_pred1[ind]))

            oof_rmse_score_lgb2 = np.sqrt(mean_squared_error(train_convert.values, trashold(oof_pred2[ind])))
            oof_cohen_score_lgb2 = qwk(train_convert.values, trashold(oof_pred2[ind]))

            oof_rmse_score = np.sqrt(mean_squared_error(train_convert,  trashold((oof_pred1[ind] + oof_pred2[ind]) / 2)))
            oof_cohen_score = qwk(train_convert.values, trashold((oof_pred1[ind] + oof_pred2[ind]) / 2))

            print('Our oof rmse score for lgb1 is:', oof_rmse_score_lgb)
            print('Our oof cohen kappa score for lbg1 is:', oof_cohen_score_lgb)
            oof_rmse_scores_lgb.append(oof_rmse_score_lgb)
            oof_cohen_scores_lgb.append(oof_cohen_score_lgb)

            print('Our oof rmse score for lgb2 is:', oof_rmse_score_lgb2)
            print('Our oof cohen kappa score for lgb2 is:', oof_cohen_score_lgb2)
            oof_rmse_scores_lgb2.append(oof_rmse_score_lgb2)
            oof_cohen_scores_lgb2.append(oof_cohen_score_lgb2)

            print('Our oof rmse score for ansamble is:', oof_rmse_score)
            print('Our oof cohen kappa score for ansamble is:', oof_cohen_score)
            oof_rmse_scores.append(oof_rmse_score)
            oof_cohen_scores.append(oof_cohen_score)

        print('Our mean rmse score for lbg1 is: ', sum(oof_rmse_scores_lgb) / len(oof_rmse_scores_lgb))
        print('Our mean cohen kappa score for lbg1 is: ', sum(oof_cohen_scores_lgb) / len(oof_cohen_scores_lgb))

        print('Our mean rmse score for lbg2 is: ', sum(oof_rmse_scores_lgb2) / len(oof_rmse_scores_lgb2))
        print('Our mean cohen kappa score for lbg2 is: ', sum(oof_cohen_scores_lgb2) / len(oof_cohen_scores_lgb2))

        print('Our mean rmse score for ansamble is: ', sum(oof_rmse_scores) / len(oof_rmse_scores))
        print('Our mean cohen kappa score for ansamble is: ', sum(oof_cohen_scores) / len(oof_cohen_scores))

    def predict(self, test):
        current_features = [x for x in self.features if x not in ['installation_id']]
        y_pred = np.zeros(len(test))
        for model in self.models:
            y_pred += np.array(model.predict(test[current_features]), dtype=float)

        return y_pred / len(self.models)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        x_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

        return -qwk(y, x_p)

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.33530578, 0.51589898, 0.71000847]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead', options={
            'maxiter': 5000})

    def predict(self, X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        return self.coef_['x']

if __name__ == '__main__':
    main()



