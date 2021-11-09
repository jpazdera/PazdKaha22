# jspsych-visual-search-circle plugin

This plugin presents a customizable visual-search task modelled after [Wang, Cavanagh, & Green (1994)](http://dx.doi.org/10.3758/BF03206946). The subject indicates whether or not a target is present among a set of distractors. The stimuli are displayed in a circle, evenly-spaced, equidistant from a fixation point. Here is an example using normal and backward Ns:

![Sample Visual Search Stimulus](/img/visual_search_example.jpg)

## Dependency

This plugin requires the Snap.svg library, available at [http://www.snapsvg.io](http://www.snapsvg.io). You must include the library in the `<head>` section of your experiment page.

## Parameters

This table lists the parameters associated with this plugin. Parameters with a default value of *undefined* must be specified. Other parameters can be left unspecified if the default value is acceptable.

Parameter | Type | Default Value | Description
----------|------|---------------|------------
target_present | boolean | *undefined* | Is the target present?
set_size | numeric | *undefined* | How many items should be displayed?
target | string | *undefined* | Path to image file that is the search target.
foil | string or array | *undefined* | Path to image file that is the foil/distractor. Can specify an array of distractors if the distractors are all different images.
fixation_image | string | *undefined* | Path to image file that is a fixation target.
target_size | array | `[50, 50]` | Two element array indicating the height and width of the search array element images.
fixation_size | array | `[16, 16]` | Two element array indicating the height and width of the fixation image.
circle_diameter | numeric | 250 | The diameter of the search array circle in pixels.
target_present_key | numeric | 74 | The key to press if the target is present in the search array.
target_absent_key | numeric | 70 | The key to press if the the target is not present in the search array.
timing_max_search | numeric | -1 | The maximum amount of time the subject is allowed to search before the trial will continue. A value of -1 will allow the subject to search indefinitely.
timing_fixation | numeric | 1000 | How long to show the fixation image for before the search array (in milliseconds).

## Data Generated

In addition to the [default data collected by all plugins](overview#datacollectedbyplugins), this plugin collects the following data for each trial.

Name | Type | Value
-----|------|------
correct | boolean | True if the subject gave the correct response.
key_press | numeric | Indicates which key the subject pressed. The value is the [numeric key code](http://www.cambiaresearch.com/articles/15/javascript-char-codes-key-codes) corresponding to the subject's response.
rt | numeric | The response time in milliseconds for the subject to make a response. The time is measured from when the stimulus first appears on the screen until the subject's response.
set_size | numeric | The number of items in the search array
target_present | boolean | True if the target is present in the search array
locations | JSON string | JSON-encoded array where each element of the array is the pixel value of the center of an image in the search array. If the target is present, then the first element will represent the location of the target.

## Example

#### Search for the backward N

```javascript
    var intro = {
      type: 'text',
      text: 'Press J if there is a backwards N. If there is no backwards N press F.'
    }

    var trial_1 = {
      target_present: true,
      set_size: 4
    }

    var trial_2 = {
      target_present: true,
      set_size: 3
    }

    var trial_3 = {
      target_present: false,
      set_size: 6
    }

    var trial_4 = {
      target_present: false,
      foil: ['img/1.gif', 'img/2.gif', 'img/3.gif'],
      set_size: 3
    }


    var trials = {
      type: 'visual-search-circle',
      target: 'img/backwardN.gif',
      foil: 'img/normalN.gif',
      fixation_image: 'img/fixation.gif',
      timeline: [trial_1, trial_2, trial_3, trial_4]
    };

    jsPsych.init({
      timeline: [intro, trials],
      on_finish: function() {
        jsPsych.data.displayData();
      }
    });
```