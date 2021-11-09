import os
import json
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages


def ltpFR3_report(stat_dir, report_dir, force=False):
    """
    Generates a report for each participant using their stats JSON file. The expected entries in the dictionary are:

    spc (Serial position curve)
    pfr (Probability of first recall)
    psr (Probability of second recall)
    ptr (Probability of third recall)
    crp_early (Conditional response probability among first three recalls)
    crp_late (Conditional response probability among recalls after the third)
    plis (Average prior list intrusions per list)
    elis (Average extra list intrusions per list)
    reps (Average number of repetitions per list)
    pli_recency (Ratio of prior list intrusions coming from each list back, up to 6 back)
    rec_per_trial (Array indicating how many words the participant correctly recalled on each trial)
    math_per_trial (Matrix indicating which math problems were answered correctly/incorrectly on each trial)
    
    And each entry contains a sub-dictionary with entries labelled with condition names, which contain that stat for
    lists of that condition only.
    
    :param stat_dir: The path to the directory where stats files are stored.
    :param report_dir: The path to the directory where new reports will be saved.
    :param force: If False, only create reports for participants who do not have them (plus the average report). If
    True, create reports for all participants. (Default == False)
    """
    stat_plotters = {'spc': plot_spc, 'pfr': plot_pfr, 'psr': plot_psr, 'ptr': plot_ptr,
                     'crp_early': plot_crp_early, 'crp_late': plot_crp_late, 'pli_recency': plot_pli_recency,
                     'elis': plot_elis, 'plis': plot_plis, 'reps': plot_reps, 'rec_per_trial': plot_rec_perlist,
                     'math_per_trial': plot_math_perlist, 'temp_fact': plot_temp_fact}

    for stat_file in glob(os.path.join(stat_dir, '*.json')):
        subj = os.path.splitext(os.path.basename(stat_file))[0]  # Get subject ID from file name
        outfile = os.path.join(report_dir, '%s.pdf' % subj)  # Define file path for stat file
        if not subj.startswith('all') and os.path.exists(outfile) and not force:  # Skip participants who already have reports
            continue

        with open(stat_file, 'r') as f:
            stats = json.load(f)

        pdf = PdfPages(outfile)
        plt.figure(figsize=(30, 30))
        if subj.startswith('all'):
            for key in stat_plotters:
                if key in stats['mean']:
                    stat_plotters[key](stats['mean'][key])
            plot_intrusions(stats['mean']['plis'], stats['mean']['elis'], stats['mean']['reps'],
                            stats['sem']['plis'], stats['sem']['elis'], stats['sem']['reps'])
            plot_elis(stats['mean']['elis'], stats['sem']['elis'])
            plot_plis(stats['mean']['plis'], stats['sem']['plis'])
            plot_reps(stats['mean']['reps'], stats['sem']['reps'])
            plot_temp_fact(stats['mean']['temp_fact'], stats['sem']['temp_fact'])
        else:
            plt.suptitle(subj, fontsize=36)
            for key in stat_plotters:

                if key in stats:
                    stat_plotters[key](stats[key])
                else:
                    print('ALERT! Missing stat %s for subject %s' % (key, subj))
            plot_intrusions(stats['plis'], stats['elis'], stats['reps'])
            # plot_elis(stats['elis'])
            # plot_plis(stats['plis'])
            # plot_reps(stats['reps'])
            # plot_temp_fact(stats['temp_fact'])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        pdf.close()
        plt.close()


def plot_spc(s):
    plt.subplot(7, 3, 1)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(1, 13), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(1, 13), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('SPC (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 13, 2), range(1, 13, 2))

    plt.subplot(7, 3, 4)
    if 'sv12' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(1, 25), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(1, 25), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('SPC (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_ffr_spc(s):
    # plt.subplot(7, 3, 1)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(1, 13), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(1, 13), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('FFR SPC (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend()
    plt.ylim(0, .3)
    plt.xticks(range(1, 13, 2), range(1, 13, 2))

    # plt.subplot(7, 3, 4)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(1, 25), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(1, 25), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('FFR SPC (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend()
    plt.ylim(0, .3)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_crp_early(s):
    plt.subplot(7, 3, 2)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(-5, 6), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(-5, 6), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(-5, 6), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(-5, 6), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('lag-CRP Early (List Length 12)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend()
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 5)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(-5, 6), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(-5, 6), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(-5, 6), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(-5, 6), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('lag-CRP Early (List Length 24)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend()
    plt.ylim(-.05, 1.05)


def plot_crp_late(s):
    plt.subplot(7, 3, 3)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(-5, 6), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(-5, 6), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(-5, 6), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(-5, 6), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('lag-CRP Late (List Length 12)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend()
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 6)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(-5, 6), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(-5, 6), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(-5, 6), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(-5, 6), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('lag-CRP Late (List Length 24)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend()
    plt.ylim(-.05, 1.05)


def plot_pfr(s):
    plt.subplot(7, 3, 7)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(1, 13), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(1, 13), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PFR (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 13, 2), range(1, 13, 2))

    plt.subplot(7, 3, 10)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(1, 25), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(1, 25), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PFR (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_psr(s):
    plt.subplot(7, 3, 8)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(1, 13), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(1, 13), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PSR (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 13, 2), range(1, 13, 2))

    plt.subplot(7, 3, 11)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(1, 25), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(1, 25), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PSR (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_ptr(s):
    plt.subplot(7, 3, 9)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(1, 13), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(1, 13), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PTR (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 13, 2), range(1, 13, 2))

    plt.subplot(7, 3, 12)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(1, 25), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(1, 25), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PTR (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend()
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_pli_recency(s):
    plt.subplot(7, 3, 13)
    if 'sv12' in s and not np.isnan(s['sv12']).all():
        plt.plot(range(1, 6), s['sv12'], 'ko-', label='Slow/Visual')
    if 'fv12' in s and not np.isnan(s['fv12']).all():
        plt.plot(range(1, 6), s['fv12'], 'k^-', label='Fast/Visual')
    if 'sa12' in s and not np.isnan(s['sa12']).all():
        plt.plot(range(1, 6), s['sa12'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa12' in s and not np.isnan(s['fa12']).all():
        plt.plot(range(1, 6), s['fa12'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PLI Recency (List Length 12)')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend()
    plt.ylim(-.05, .55)

    plt.subplot(7, 3, 16)
    if 'sv24' in s and not np.isnan(s['sv24']).all():
        plt.plot(range(1, 6), s['sv24'], 'ko-', label='Slow/Visual')
    if 'fv24' in s and not np.isnan(s['fv24']).all():
        plt.plot(range(1, 6), s['fv24'], 'k^-', label='Fast/Visual')
    if 'sa24' in s and not np.isnan(s['sa24']).all():
        plt.plot(range(1, 6), s['sa24'], 'ko--', markerfacecolor='white', label='Slow/Auditory')
    if 'fa24' in s and not np.isnan(s['fa24']).all():
        plt.plot(range(1, 6), s['fa24'], 'k^--', markerfacecolor='white', label='Fast/Auditory')
    plt.title('PLI Recency (List Length 24)')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend()
    plt.ylim(-.05, .55)


def plot_intrusions(plis, elis, reps, pli_err=None, eli_err=None, rep_err=None):
    plt.subplot(7, 3, 14)
    if None in (pli_err, eli_err, rep_err):
        plt.bar([1, 2, 3], [plis['all'], elis['all'], reps['all']], align='center', color='k', fill=False)
    else:
        pli_err = 1.96 * pli_err['all']
        eli_err = 1.96 * eli_err['all']
        rep_err = 1.96 * rep_err['all']
        plt.bar([1, 2, 3], [plis['all'], elis['all'], reps['all']], yerr=[pli_err, eli_err, rep_err], align='center',
                color='k', fill=False)
    plt.xticks([1, 2, 3], ['PLI', 'ELI', 'Rep'])
    plt.title('Intrusions')
    plt.ylabel('Intrusions Per List')


def plot_elis(s, err=None):
    plt.subplot(7, 3, 15)
    if err is None:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-', label='Visual')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white', label='Auditory')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12']*1.96, err['fv12']*1.96], fmt='ko-')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12']*1.96, err['fa12']*1.96], fmt='ko--', markerfacecolor='white')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24']*1.96, err['fv24']*1.96], fmt='ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24']*1.96, err['fa24']*1.96], fmt='ko--', markerfacecolor='white')
        plt.ylim(0, 1.05)
    plt.title('ELIs')
    plt.ylabel('ELIs Per List')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend()


def plot_plis(s, err=None):
    plt.subplot(7, 3, 17)
    if err is None:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-', label='Visual')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white', label='Auditory')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12']*1.96, err['fv12']*1.96], fmt='ko-')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12']*1.96, err['fa12']*1.96], fmt='ko--', markerfacecolor='white')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24']*1.96, err['fv24']*1.96], fmt='ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24']*1.96, err['fa24']*1.96], fmt='ko--', markerfacecolor='white')
        plt.ylim(0, .45)
    plt.title('PLIs')
    plt.ylabel('PLIs Per List')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend()


def plot_reps(s, err=None):
    plt.subplot(7, 3, 18)
    if err is None:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-', label='Visual')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white', label='Auditory')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12']*1.96, err['fv12']*1.96], fmt='ko-')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12']*1.96, err['fa12']*1.96], fmt='ko--', markerfacecolor='white')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24']*1.96, err['fv24']*1.96], fmt='ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24']*1.96, err['fa24']*1.96], fmt='ko--', markerfacecolor='white')
        plt.ylim(0, .45)
    plt.title('Repetitions')
    plt.ylabel('Reps Per List')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend()


def plot_temp_fact(s, err=None):
    plt.subplot(7, 3, 20)
    if err is None:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-', label='Visual')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white', label='Auditory')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        if 'sv12' in s and not np.isnan(s['sv12']).all() and 'fv12' in s and not np.isnan(s['fv12']).all():
            plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12']*1.96, err['fv12']*1.96], fmt='ko-')
        if 'sa12' in s and not np.isnan(s['sa12']).all() and 'fa12' in s and not np.isnan(s['fa12']).all():
            plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12']*1.96, err['fa12']*1.96], fmt='ko--', markerfacecolor='white')
        if 'sv24' in s and not np.isnan(s['sv24']).all() and 'fv24' in s and not np.isnan(s['fv24']).all():
            plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24']*1.96, err['fv24']*1.96], fmt='ko-')
        if 'sa24' in s and not np.isnan(s['sa24']).all() and 'fa24' in s and not np.isnan(s['fa24']).all():
            plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24']*1.96, err['fa24']*1.96], fmt='ko--', markerfacecolor='white')
    plt.title('Temporal Clustering Factor')
    plt.ylabel('Clustering Score')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend()
    plt.ylim(.5, .8)


def plot_rec_perlist(s):
    plt.subplot(7, 3, 19)
    plt.plot(range(1, 19), s, 'ko-')
    plt.title('Recall Performance Check')
    plt.xlabel('Trial Number')
    plt.ylabel('Probability of Recall')
    plt.xlim(0, 19)
    plt.ylim(-.05, 1.05)


def plot_math_perlist(s):
    plt.subplot(7, 3, 21)
    plt.plot(range(1, 19), s, 'ko-')
    plt.title('Math Performance Check')
    plt.xlabel('Trial Number')
    plt.ylabel('Math Correct')
    plt.xlim(0, 19)
    plt.ylim(-.5, 20.5)
