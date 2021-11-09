jsPsych.plugins['math-distractor'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'math-distractor',
    description: ''
  }

  plugin.trial = function(display_element, trial) {

    // Default value for time limit option
    trial.duration = trial.duration || -1;
    // Time handlers
    var setTimeoutHandlers = [];

    // if any trial variables are functions
    // this evaluates the function and replaces
    // it with the output of the function
    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);

    var rts = [];
    var responses = [];
    var num_a = [];
    var num_b = [];
    var num_c = [];
    var key_presses = [];
    var key_times = [];
    var tbox = '<textarea id="math_box"></textarea></div>'

    var gen_trial = function() {
      // setup question and response box
      var nums = [randomInt(1,10), randomInt(1,10), randomInt(1,10)];
      var prob = '<div id="stim">' + nums[0].toString() + ' + ' + nums[1].toString() + ' + ' + nums[2].toString() + ' = ';
      display_element.innerHTML = prob + tbox;

      // log the new problem
      num_a.push(nums[0]);
      num_b.push(nums[1]);
      num_c.push(nums[2]);

      // set up response collection
      $('textarea').keydown(function(e){
        // Get timing of key press relative to start of math task
        var endTime = (new Date()).getTime();
        var response_time = endTime - startTime;
        // Record key press and its timing
        key_presses.push(e.keyCode)
        key_times.push(response_time)
        // if enter is pressed while in textarea
        if (e.keyCode===13) {
          // get response time (when participant presses enter)
          rts.push(response_time);
          // get submitted answer
          var ans = display_element.querySelector('textarea').value;
          responses.push(ans);
          // generate new problem
          gen_trial();
          // suppress the newline character
          return false;
        }
      });

      // automatically place cursor in textarea when page loads
      $(function(){
        $('textarea').focus();
      });
    };

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
        "responses": responses,
        "num1": num_a,
        "num2": num_b,
        "num3": num_c,
        "key_presses": key_presses,
        "key_times": key_times
      };

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };
    gen_trial();

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
