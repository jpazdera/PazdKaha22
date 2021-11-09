import os


def sort_excluded_files():
    with open('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXCLUDED.txt', 'r') as f:
        exc = [s.strip() for s in f.readlines()]

    with open('/data/eeg/scalp/ltp/ltpFR3_MTurk/BAD_SESS.txt', 'r') as f:
        bad_sess = [s.strip() for s in f.readlines()]

    with open('/data/eeg/scalp/ltp/ltpFR3_MTurk/REJECTED.txt', 'r') as f:
        rej = [s.strip() for s in f.readlines()]

    for subj in exc:
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/events/excluded/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/data/excluded/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/excluded/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/%s.pdf' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/%s.pdf' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/excluded/%s.pdf' % subj)

    for subj in bad_sess:
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/events/bad_sess/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/data/bad_sess/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/bad_sess/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/%s.pdf' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/%s.pdf' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/bad_sess/%s.pdf' % subj)

    for subj in rej:
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/events/rejected/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/data/rejected/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/%s.json' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/%s.json' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/rejected/%s.json' % subj)
        if os.path.exists('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/%s.pdf' % subj):
            os.rename('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/%s.pdf' % subj, '/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/rejected/%s.pdf' % subj)


if __name__ == "__main__":
    sort_excluded_files()
