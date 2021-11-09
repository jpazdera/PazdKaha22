var counter = [];
for (var i = 10; i > 0; i--) {
  counter.push({stimulus: '<p id="stim">'.concat(i.toString(), '</p>')});
}

var countdown = {
  type: 'single-stim',
  timing_response: 1000,
  response_ends_trial: false,
  is_html: true,
  timeline: counter
};
