import csv
import json
import numpy as np
from glob import glob

def process_survey(outfile):
    """

    :param outfile:
    :return:
    """
    # Load existing survey spreadsheet
    # Note that this assumes the data from MTK0001-MTK0224 who took the separate questionnaire has been already saved
    with open(outfile, 'r') as f:
        r = csv.reader(f, delimiter=',')
        s = [row for row in r][1:]

    with open('/data/eeg/scalp/ltp/ltpFR3_MTurk/VERSION_STARTS.json', 'r') as f:
        exp2_start = json.load(f)['2']

    # Get locations of all session log files
    log_files_good = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/MTK[0-9][0-9][0-9][0-9].json')
    log_files_excluded = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/excluded/MTK[0-9][0-9][0-9][0-9].json')
    log_files_bad_sess = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/bad_sess/MTK[0-9][0-9][0-9][0-9].json')
    log_files_all = log_files_good + log_files_excluded + log_files_bad_sess

    # Get list of participants whose data has already been processed
    already_processed = [row[0] for row in s]

    # Process the responses from each new log file
    for lf in log_files_all:
        subj = lf[-12:-5]
        if subj in already_processed:
            continue
        print(subj)

        # Load questionnaire data for participant, as well as auditory/visual condition for experiment 2
        with open(lf, 'r') as f:
            data = json.load(f)
            cond = data['condition']
            data = data['questiondata']

        # Extract demographic info
        age = data['age'] if 'age' in data else ''
        education = data['education'] if 'education' in data else 'Not Reported'
        ethnicity = data['ethnicity'] if 'ethnicity' in data else 'Not Reported'
        gender = data['gender'] if 'gender' in data else 'Not Reported'
        gender_other = data['gender_other'] if 'gender_other' in data else ''
        language = data['language'] if 'language' in data else 'Not Reported'
        marital = data['marital'] if 'marital' in data else 'Not Reported'
        origin = data['origin'] if 'origin' in data else 'Not Reported'
        race = '|'.join(data['race']) if 'race' in data else 'Not Reported'
        race_other = data['race_other'] if 'race_other' in data else ''

        # Extract exit survey info from experiment 1
        if int(subj[-4:]) < exp2_start:
            strat_categorize_aud = data['strat-aud-categorize'] if 'strat-aud-categorize' in data else ''
            strat_categorize_vis = data['strat-vis-categorize'] if 'strat-vis-categorize' in data else ''
            strat_image_aud = data['strat-aud-image'] if 'strat-aud-image' in data else ''
            strat_image_vis = data['strat-vis-image'] if 'strat-vis-image' in data else ''
            strat_rehearsal_aud = data['strat-aud-rehearsal'] if 'strat-aud-rehearsal' in data else ''
            strat_rehearsal_vis = data['strat-vis-rehearsal'] if 'strat-vis-rehearsal' in data else ''
            strat_spatial_aud = data['strat-aud-spatial'] if 'strat-aud-spatial' in data else ''
            strat_spatial_vis = data['strat-vis-spatial'] if 'strat-vis-spatial' in data else ''
            strat_story_aud = data['strat-aud-story'] if 'strat-aud-story' in data else ''
            strat_story_vis = data['strat-vis-story'] if 'strat-vis-story' in data else ''

        # Extract exit survey info from experiment 2 visual participants
        elif cond == 0:
            strat_categorize_aud = ''
            strat_categorize_vis = data['strat-categorize'] if 'strat-categorize' in data else ''
            strat_image_aud = ''
            strat_image_vis = data['strat-image'] if 'strat-image' in data else ''
            strat_rehearsal_aud = ''
            strat_rehearsal_vis = data['strat-rehearsal'] if 'strat-rehearsal' in data else ''
            strat_spatial_aud = ''
            strat_spatial_vis = data['strat-spatial'] if 'strat-spatial' in data else ''
            strat_story_aud = ''
            strat_story_vis = data['strat-story'] if 'strat-story' in data else ''

        # Extract exit survey info from experiment 2 auditory participants
        else:
            strat_categorize_aud = data['strat-categorize'] if 'strat-categorize' in data else ''
            strat_categorize_vis = ''
            strat_image_aud = data['strat-image'] if 'strat-image' in data else ''
            strat_image_vis = ''
            strat_rehearsal_aud = data['strat-rehearsal'] if 'strat-rehearsal' in data else ''
            strat_rehearsal_vis = ''
            strat_spatial_aud = data['strat-spatial'] if 'strat-spatial' in data else ''
            strat_spatial_vis = ''
            strat_story_aud = data['strat-story'] if 'strat-story' in data else ''
            strat_story_vis = ''

        strat_other = data['strat-other'] if 'strat-other' in data else ''

        if 'wrote-notes' in data:
            if data['wrote-notes'] == 'Yes':
                wrote_notes = '1'
            elif data['wrote-notes'] == 'No':
                wrote_notes = '0'
            else:
                wrote_notes = ''
        else:
            wrote_notes = ''

        if 'distracted' in data:
            if data['distracted'] == 'Yes':
                distracted = '1'
            elif data['distracted'] == 'No':
                distracted = '0'
            else:
                distracted = ''
        else:
            distracted = ''

        # Add new row to spreadsheet
        s.append([subj, age, education, ethnicity, gender, gender_other, language, marital, origin, race, race_other,
                  strat_categorize_aud, strat_categorize_vis, strat_image_aud, strat_image_vis, strat_rehearsal_aud,
                  strat_rehearsal_vis, strat_spatial_aud, strat_spatial_vis, strat_story_aud, strat_story_vis,
                  strat_other, wrote_notes, distracted])

        head = ['subject', 'age', 'education', 'ethnicity', 'gender', 'gender_other', 'language', 'marital', 'origin',
                'race', 'race_other', 'strat_categorize_aud', 'strat_categorize_vis', 'strat_image_aud',
                'strat_image_vis', 'strat_rehearsal_aud', 'strat_rehearsal_vis', 'strat_spatial_aud',
                'strat_spatial_vis', 'strat_story_aud', 'strat_story_vis', 'strat_other', 'wrote_notes', 'distracted']

        # Sort by subject ID
        s.sort(key=lambda x: x[0])

        # Write data out to file
        with open(outfile, 'w') as f:
            w = csv.writer(f, delimiter=',')
            w.writerow(head)
            for row in s:
                w.writerow(row)

    # Write out WROTE_NOTES.txt file
    all_subj = np.array(s)[:, 0]
    wn = np.array(s)[:, -2]
    subj_wrote_notes = all_subj[wn == '1']
    np.savetxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/WROTE_NOTES.txt', subj_wrote_notes, fmt='%s')