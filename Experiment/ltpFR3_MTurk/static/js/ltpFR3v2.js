/* - - - - PSITURK - - - - */
var psiturk = new PsiTurk(uniqueId, adServerLoc, mode);

// Record screen resolution & available screen size
psiturk.recordUnstructuredData('screen_width', screen.width)
psiturk.recordUnstructuredData('screen_height', screen.height)
psiturk.recordUnstructuredData('avail_screen_width', screen.availWidth)
psiturk.recordUnstructuredData('avail_screen_height', screen.availHeight)
psiturk.recordUnstructuredData('color_depth', screen.colorDepth)
psiturk.recordUnstructuredData('pixel_depth', screen.pixelDepth)

var pages = [
	"survey2.html"
];

psiturk.preloadPages(pages);

var Questionnaire = function() {
	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";
	record_responses = function() {
		psiturk.recordTrialData({'phase':'survey', 'status':'submit'});
		$('input[type=radio]:checked').each( function(i, val) {
			psiturk.recordUnstructuredData(this.name, this.value);
		});
		$('input[type=number]').each( function(i, val) {
			psiturk.recordUnstructuredData(this.name, this.value);
		});
		$('input[type=text]').each( function(i, val) {
			psiturk.recordUnstructuredData(this.name, this.value);
		});
		$('input[type=range]').each( function(i, val) {
			psiturk.recordUnstructuredData(this.name, this.value);
		});
		$('select').each( function(i, name) {
			psiturk.recordUnstructuredData(this.name, this.value);
		});
		$('textarea').each( function(i, name) {
			psiturk.recordUnstructuredData(this.name, this.value);
		});
		var races = $('input[type=checkbox]:checked').map(function () {
    	return this.value;
		}).get();
		psiturk.recordUnstructuredData('race', races);
	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	/*resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);
		psiturk.saveData({
			success: function() {
			    clearInterval(reprompt);
					psiturk.completeHIT(); // when finished saving compute bonus, the quit
    	},
			error: prompt_resubmit});
	};*/

	// Load the questionnaire snippet
	psiturk.showPage('survey2.html');
	psiturk.recordTrialData({'phase':'survey', 'status':'begin'});

	//$(document).on('submit', form.postquiz, function() {
	$("#postquiz").submit(function() {
			// Turn off pop-up warning for closing browser so it doesn't try to block the experiment from closing
      window.onbeforeunload = null;
	    record_responses();
			saveData(JSON.stringify(psiturk.taskdata.toJSON()));
			psiturk.completeHIT();
	    /*psiturk.saveData({
            success: function(){
                	psiturk.completeHIT(); // when finished saving, mark HIT complete and quit
            },
            error: prompt_resubmit});*/
	});
	/*$("#next").click(function () {
      // Turn off pop-up warning for closing browser so it doesn't try to block the experiment from closing
      window.onbeforeunload = null;
	    record_responses();
	    psiturk.saveData({
            success: function(){
                	psiturk.completeHIT(); // when finished saving, mark HIT complete and quit
            },
            error: prompt_resubmit});
	});*/
};

/* - - - - SETTINGS - - - - */

var isi = 800; // Inter-stimulus interval
var isi_jitter = 400; // Jitter on the ISI
var rec_dur = 60000; // Duration of recall periods
var ffr_dur = 300000; // Duration of final free recall
var pre_recall_delay = 1200; // Extra delay after the final word in a list
var pre_recall_jitter = 200; // Jitter on the pre-recall delay
var post_trial_delay = 2000; // Delay between the end of trial screen and the start of distractor
var post_dist_delay = 2000; // Delay following each distractor period
var post_countdown_delay = 1500; // Delay between the pre-list countdown and first presentation

/* - - - - INSTRUCTIONS - - - - */

var welcome_block = {
  type: 'text',
  text: '<p id="inst">Welcome to the experiment. Press any key to begin.</p>',
  data: {type: 'INSTRUCTIONS'}
};

var audio_test_intro = {
	type: 'text',
	text: '<p id="inst">Due to the use of auditory stimuli in this task, it is important that you have the volume on your computer turned on and set loud enough to hear the words and sounds that we may be presenting. In order to ensure that you will be able to perform the auditory portions of the task, we ask that you begin by completing the following audio test.</p><p id="inst">You will see a series of three pages, each containing a play button and an empty textbox. On each page, click the play button to listen to an audio clip of a single word, then type that word into the textbox and hit ENTER to proceed. You will be able to replay the word as many times as needed, giving you the chance to adjust your volume to an appropriate level. Press any key to begin the audio test.</p>',
	data: {type: 'INSTRUCTIONS'}
};

if (condition == 0) {
	var instructions = {
	  type: 'text',
		timing_post_trial: 500,
	  text: '<p id="inst">During the course of this study, you will see lists of words, which you will be asked to remember. Each set of words will be presented visually on-screen, one at a time. As you see each word, try to focus on that word alone, without thinking back to previous words. After each series of words ends, a tone will sound and a row of asterisks will flash on the screen, and you will see a screen with an empty textbox appear.<p id="inst">When the textbox appears, try to recall as many words as possible from the most recent list. As you recall each word, type it into the provided textbox. Pressing ENTER, SPACE, COMMA, or SEMICOLON will submit the word and clear the textbox. Please type words into the box one at a time, submitting each as you go.</p><p id="inst">You do not need to worry about capitalization while entering your words. If you are unsure of how to spell a word, simply type it to the best of your ability. You will be given a fixed amount of time to recall the words. Please try hard throughout the recall period, as you may recall some words even when you feel that you have exhausted your memory.</p><p id="inst">Press any key to continue reading the instructions...</p>',
	  data: {type: 'INSTRUCTIONS'}
	};
} else {
	var instructions = {
	  type: 'text',
		timing_post_trial: 500,
	  text: '<p id="inst">During the course of this study, you will hear lists of words, which you will be asked to remember. Each set of words will be presented as a sequence of audio clips. As you hear each word, try to focus on that word alone, without thinking back to previous words. After each series of words ends, a tone will sound and a row of asterisks will flash on the screen, and you will see a screen with an empty textbox appear.<p id="inst">When the textbox appears, try to recall as many words as possible from the most recent list. As you recall each word, type it into the provided textbox. Pressing ENTER, SPACE, COMMA, or SEMICOLON will submit the word and clear the textbox. Please type words into the box one at a time, submitting each as you go.</p><p id="inst">You do not need to worry about capitalization while entering your words. If you are unsure of how to spell a word, simply type it to the best of your ability. You will be given a fixed amount of time to recall the words. Please try hard throughout the recall period, as you may recall some words even when you feel that you have exhausted your memory.</p><p id="inst">Press any key to continue reading the instructions...</p>',
	  data: {type: 'INSTRUCTIONS'}
	};
}

var instructions_math = {
  type: 'text',
	timing_post_trial: 500,
  text: '<p id="inst">Before each list of words, you will be given a series of math problems of the form A+B+C, where A, B, and C are small integers. Try to answer the math problems as quickly and accurately as possible. For each problem, type your answer into the provided textbox and hit ENTER to submit it. Following the math problems, you will see a 10-second countdown appear on-screen, warning you of the start of the word list. Once the countdown has ended, presentation of the word list will begin.</p><p id="inst">Press any key to continue reading the instructions...</p>',
  data: {type: 'INSTRUCTIONS'}
};

var instructions_practice = {
  type: 'text',
  text: '<p id="inst">You will now have the chance to familiarize yourself with the task by completing two practice trials. Remember that there are three steps in each trial. First, you will be asked to complete a series of math problems of the form A+B+C. Then, you will be presented with a series of words, one at a time. Finally, you will be asked to recall the words in any order, entering each into the provided textbox.</p><p id="inst">Press any key to begin the practice trials.</p>',
  timing_post_trial: 1000,
  data: {type: 'INSTRUCTIONS'}
};

var instructions_end_practice = {
  type: 'text',
  text: '<p id="inst">You have completed the practice trials.</p><p id="inst">Press any key when you are ready to begin the main phase of the experiment.</p>',
  timing_post_trial: 1000,
  data: {type: 'INSTRUCTIONS'}
};

var trial_end_screen = {
  type: 'text',
  timing_post_trial: post_trial_delay,
  text: '<p id="inst">Trial complete. Press any key to continue.</p>',
  data: {type: 'INSTRUCTIONS'}
}

var instructions_ffr = {
    type: 'text',
    timing_post_trial: 5000,
    text: '<p id="inst">You will now be given one final recall test. During this next period, please recall as many items as you can remember from the entire session (including practice lists), in any order. You will be given several minutes to do this. Try hard to recall for the entire period, as you may find that words continue to come to you, even when you feel that you have exhausted your memory.</p><p id="inst">Press any key when you are ready to continue.</p>',
    data: {type: 'INSTRUCTIONS'}
};

/* Old ending screen (replaced by exit survey)
var ending = {
  type: 'text',
  text: '<p id="inst">You have now completed the study.</p><p id="inst">Thank you for participating! If you are interested in learning more about our lab and the work that we do, we invite you to visit our website at memory.psych.upenn.edu after submitting your work.</p><p id="inst">Press any key to mark your HIT as complete and return to MTurk. There, you should see a screen with a green button labeled "Complete HIT". Press this button to submit your HIT for approval.</p>',
  data: {type: 'INSTRUCTIONS'}
};
*/

/* - - - - AUDIOTEST - - - - */

var audio_test1 = {
	type:'audio-test',
	preamble: ['<h1>Word 1</h1></p><p id="inst">Click the play button to listen to the word, then enter the word you hear into the textbox below.</p><audio controls preload="auto"><source src="/static/audio/wordpool/AudioTest/Test1.wav" type="audio/wav">Your browser does not support the audio element.</audio>'],
	questions: [''],
	word: 'ocean',
	data: {type: 'AUDIO_TEST'}
};

var audio_test2 = {
	type:'audio-test',
	preamble: ['<h1>Word 2</h1></p><p id="inst">Click the play button to listen to the word, then enter the word you hear into the textbox below.</p><audio controls preload="none"><source src="/static/audio/wordpool/AudioTest/Test2.wav" type="audio/wav">Your browser does not support the audio element.</audio>'],
	questions: [''],
	word: 'crystal',
	data: {type: 'AUDIO_TEST'}
};

var audio_test3 = {
	type:'audio-test',
	preamble: ['<h1>Word 3</h1><p id="inst">Click the play button to listen to the word, then enter the word you hear into the textbox below.</p><audio controls preload="none"><source src="/static/audio/wordpool/AudioTest/Test3.wav" type="audio/wav">Your browser does not support the audio element.</audio>'],
	questions: [''],
	word: 'spice',
	timing_post_trial: 750,
	data: {type: 'AUDIO_TEST'}
};

/* - - - - SLEEP SURVEY - - - - */

var sleep_survey = {
	type: 'survey-text',
	timing_post_trial: 1000,
	preamble: '<p id="inst">Thank you for completing the audio test. We appreciate you taking the time to check your sound before we begin.</p><p id="inst">Please answer the following questions. Then, you will be introduced to the main task.</p>',
	questions: ['What time did you fall asleep last night?', 'What time did you wake up?', 'On a scale of 1 (Not at all) to 5 (Very), how alert do you feel right now?'],
	data: {type: 'SLEEP_SURVEY'}
}

/* - - - - UTILITY - - - - */

var fixation = {
  type: 'single-stim',
  timing_response: post_countdown_delay,
  response_ends_trial: false,
  is_html: true,
  timeline: [{stimulus: '<p id="stim">+</p>'}],
  data: {type: 'FIXATION'}
};

var prerec_alert = {
  type: 'single-audio',
  timing_response: 500,
  response_ends_trial: false,
  stimulus: '/static/audio/400Hz.wav',
  prompt: '<p id="stim">*****</p>',
  data: {type: 'PRE_RECALL_ALERT'}
};

/* - - - - STIMULI - - - - */

var lists;
var conditions;

// Each trial has one entry, which is a 4-item list identifying the conditions
// for that list, as follows:
// conditions[i][0] = list length
// conditions[i][1] = presentation rate
// conditions[i][2] = modality
// conditions[i][3] = distractor duration
conditions = sess_info.conditions[0];
// For v2 (between-subjects modality), set all modalities to same modality
// Condition 0 = visual, Condition 1 = auditory
if (condition == 0) {
	for (i=0; i < conditions.length; i++) {
		conditions[i][2] = 'v'
	}
} else if (condition == 1) {
	for (i=0; i < conditions.length; i++) {
		conditions[i][2] = 'a'
	}
}

// Each list is represented by a list of the words to be presented, in order
lists = sess_info.word_order[0];

var i;
var j;

// Randomize the interstimulus intervals for each list
var intervals = [];
var list_isi;
for (i = 0; i < conditions.length; i++) {
  list_isi = [];
  for (j = 0; j < conditions[i][0]; j++) {  // conditions[i][0] is list length
    list_isi.push(Math.floor(isi + Math.random() * isi_jitter));
  }
  intervals.push(list_isi);
}

// Generate word lists based on the list lengths and presentation rates
var all_lists = [];
var l;
for (i = 0; i < conditions.length; i++) {
  l = [];
  // Generate a list with visual presentation
  if (conditions[i][2] == 'v') {
    for (j = 0; j < conditions[i][0]; j++) {
      l.push({
        type: 'single-stim',
        timing_response: conditions[i][1],
        timing_post_trial: Math.floor(isi + Math.random() * isi_jitter),
        response_ends_trial: false,
        is_html: true,
        stimulus: '<p id="stim">'.concat(lists[i][j], '</p>'),
        data: {type: 'PRES_VIS', word: lists[i][j].toLowerCase(), wordID: wordpool.indexOf(lists[i][j]), conditions: conditions[i], trial: i, serialpos: j}
      });
    }
  // Generate a list with auditory presentation
  } else if (conditions[i][2] == 'a') {
    for (j = 0; j < conditions[i][0]; j++) {
      l.push({
        type: 'single-audio',
        timing_response: conditions[i][1] + Math.floor(isi + Math.random() * isi_jitter),
        response_ends_trial: false,
        is_html: true,
        stimulus: '/static/audio/wordpool/'.concat(lists[i][j], '.wav'),
        data: {type: 'PRES_AUD', word: lists[i][j].toLowerCase(), wordID: wordpool.indexOf(lists[i][j]), conditions: conditions[i], trial: i, serialpos: j}
      });
    }
  }
  all_lists.push(l);
}

/* - - - - FREE RECALL - - - - */

var free_recall_trial = [{
  preamble: [''],
  questions: [''],
  on_finish: function() {
    //psiturk.saveData();
		saveData(JSON.stringify(psiturk.taskdata.toJSON()));
  }
}];

var final_free_recall = {
  type: 'free-recall',
  duration: ffr_dur,
  timeline: free_recall_trial,
  data: {type: 'FFR'},
	timing_post_trial: post_trial_delay,
  on_finish: function() {
    //psiturk.saveData();
		saveData(JSON.stringify(psiturk.taskdata.toJSON()));
  }
};

/* - - - - BLOCKING - - - - */

var timeline_all = [welcome_block, audio_test_intro, audio_test1, audio_test2, audio_test3, sleep_survey, instructions, instructions_math, instructions_practice];
for (i = 0; i < all_lists.length; i++) {
//for (i = 0; i < 2; i++) {
  timeline_all = timeline_all.concat({type: 'math-distractor', timing_post_trial: post_dist_delay, duration: conditions[i][3], data: {type:'DISTRACTOR', conditions: conditions[i], trial: i}}, countdown, fixation, all_lists[i], {type: 'single-stim', timing_response: Math.floor(pre_recall_delay + Math.random() * pre_recall_jitter), data: {type: 'PRE_RECALL_DELAY'}, response_ends_trial: false, is_html: true, timeline: [{stimulus: ''}]}, prerec_alert, {type: 'free-recall', duration: rec_dur, timing_post_trial: post_trial_delay, timeline: free_recall_trial, data: {type: 'FREE_RECALL', conditions: conditions[i], trial: i}});
  if (i == 1) {
    // Add end-practice instructions after the second practice list
    timeline_all.push(instructions_end_practice);
  } else {
    timeline_all.push(trial_end_screen);
  }
}
timeline_all = timeline_all.concat(instructions_ffr, prerec_alert, final_free_recall);
//timeline_all = [welcome_block];
window.onbeforeunload = function() {
    return "Warning: Refreshing the window will RESTART the experiment from the beginning! Please avoid refreshing your browser while the task is running.";
}

/* - - - - EXECUTION - - - - */

jsPsych.init({
  timeline: timeline_all,
  on_finish: function() {
    //psiturk.saveData();
		saveData(JSON.stringify(psiturk.taskdata.toJSON()));
    new Questionnaire();
		// HIT completion has been moved to the questionnaire's
    /*psiTurk.saveData({
          success: function(){
                psiTurk.completeHIT();
          },
          error: prompt_resubmit});*/
  },
  on_data_update: function(data) {
    psiturk.recordTrialData(data);
  }
});
