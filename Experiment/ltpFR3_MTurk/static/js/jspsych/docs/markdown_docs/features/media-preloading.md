# Media Preloading

If an experiment uses images or audio files as stimuli, it is important to preload the files before running the experiment. Preloading files means that the subject's browser will download all of the files and store them in local memory on the subject's computer. This is important because loading an image file is much faster if it is already in memory on the subject's computer. Without preloading, there will be noticeable delays in the display of media, which will affect any timing measurements (such as how long an image is displayed, or a subject's response time since first viewing an image).

jsPsych will automatically preload audio and image files that are used as parameters for the standard set of plugins.

```javascript
// the image file img/file1.png is
// automatically preloaded before the experiment begins
var trial = {
	type: 'single-stim',
	stimulus: 'img/file1.png'
}

// the sound file is also preloaded automatically
var sound_trial = {
	type: 'single-audio',
	stimulus: 'snd/hello.mp3'
}

jsPsych.init({
	timeline: [trial]
});
```

If you are using images or audio in your experiment but they are not being passed directly as parameters to the trials (e.g., because you are using functions as parameters that return the image or audio), then you may need to manually preload the image files.

You can specify an array of image files and an array of audio files for preloading in the `jsPsych.init()` method. These files will load before the experiment starts.

```javascript
// this trial will not preload the images, because the image file is being used
// in an HTML string
var trial = {
	type: 'single-stim',
	stimulus: '<img src="img/file1.png"></img>',
	is_html: true
}

var audio_trial = {
	type: 'single-audio',
	stimulus: function() { return 'audio/foo.mp3' }
}

// an array of paths to images that need to be loaded
var images = ['img/file1.png'];
var audio = ['audio/foo.mp3'];

jsPsych.init({
	timeline: [trial],
	audio_preload: audio,
	image_preload: images
});

```
