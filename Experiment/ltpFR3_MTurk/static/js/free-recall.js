jsPsych.plugins['free-recall'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'free-recall',
    description: '',
    parameters: {
      questions: {
        type: [jsPsych.plugins.parameterType.STRING],
        array: true,
        default: undefined,
        no_function: false,
        description: ''
      },
      premable: {
        type: [jsPsych.plugins.parameterType.STRING],
        default: '',
        no_function: false,
        description: ''
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    trial.preamble = typeof trial.preamble == 'undefined' ? "" : trial.preamble;
    if (typeof trial.rows == 'undefined') {
      trial.rows = [];
      for (var i = 0; i < trial.questions.length; i++) {
        trial.rows.push(1);
      }
    }
    if (typeof trial.columns == 'undefined') {
      trial.columns = [];
      for (var i = 0; i < trial.questions.length; i++) {
        trial.columns.push(40);
      }
    }

    // Default value for time limit option
    trial.duration = trial.duration || -1;
    // Time handlers
    var setTimeoutHandlers = [];

    // if any trial variables are functions
    // this evaluates the function and replaces
    // it with the output of the function
    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);

    // show preamble text
    display_element.innerHTML += '<div id="jspsych-free-recall-preamble" class="jspsych-free-recall-preamble">'+trial.preamble+'</div>';

    // add question and textbox for answer
    display_element.innerHTML += '<div id="jspsych-free-recall" class="jspsych-free-recall-question" style="margin: 2em 0em;">'+
      '<p class="jspsych-free-recall">' + trial.questions + '</p>'+
      '<textarea name="#jspsych-free-recall-response" id="recall_box" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false"></textarea>'+
      '</div>';

    // set up response collection
    var rts = [];
    var recalled_words = [];
    var key_presses = [];
    var key_times = [];
    $('textarea').keydown(function(e){
      // Get timing of key press relative to start of recall period
      var endTime = (new Date()).getTime();
      var response_time = endTime - startTime;
      // Record key press and its timing
      key_presses.push(e.keyCode)
      key_times.push(response_time)
      // If enter, space, semicolon, or comma is pressed, record the word and
      // its timing, then clear the text box for the next word
      if (e.keyCode===13 | e.keyCode===32 | e.keyCode===186 | e.keyCode===188) {
        rts.push(response_time);
        // get recalled word
        word = display_element.querySelector('textarea').value.toLowerCase();
        recalled_words.push(word);
        // empty the contents of the textarea
        display_element.querySelector('textarea').value = '';
        // suppress the character that was entered
        return false;
      }
    });

    // automatically place cursor in textarea when page loads
    $(function(){
      $('textarea').focus();
    });

    var end_trial = function() {
      // kill any remaining setTimeout handlers
      for (var i = 0; i < setTimeoutHandlers.length; i++) {
        clearTimeout(setTimeoutHandlers[i]);
      }

      // kill keyboard listeners
      if (typeof keyboardListener !== 'undefined') {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
      }

      // clear the display
      display_element.innerHTML = '';

      // gather the data to store for the trial
      var trial_data = {
        "rt": rts,
        "recwords": recalled_words,
        "key_presses": key_presses,
        "key_times": key_times
      };

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    var startTime = (new Date()).getTime();

    if (trial.duration > 0) {
      var t2 = setTimeout(function() {
        end_trial();
      }, trial.duration);
      setTimeoutHandlers.push(t2);
    }

  };


  return plugin;
})();
