// from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random
function randomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
}

function loadListsAndRun() {
  // Loads the list data for the specified subject number by adding it as a
  // source on the page
  var head = document.getElementsByTagName('head')[0];
  var js = document.createElement("script");
  js.type = "text/javascript";
  js.src = "/static/pools/lists/".concat(counterbalance, ".js");
  js.async = false;
  head.appendChild(js);
  // Make sure ltpFR3.js is run AFTER we load the session info
  var js2 = document.createElement("script");
  js2.type = "text/javascript";
  js2.src = "/static/js/ltpFR3.js";
  js2.async = false;
  head.appendChild(js2);
}

function loadListsAndRunV2() {
  // Loads the list data for the specified subject number by adding it as a
  // source on the page
  var head = document.getElementsByTagName('head')[0];
  var js = document.createElement("script");
  js.type = "text/javascript";
  js.src = "/static/pools/lists/".concat(counterbalance, ".js");
  js.async = false;
  head.appendChild(js);
  // Make sure ltpFR3v2.js is run AFTER we load the session info
  var js2 = document.createElement("script");
  js2.type = "text/javascript";
  js2.src = "/static/js/ltpFR3v2.js";
  js2.async = false;
  head.appendChild(js2);
}

function saveData(filedata) {
  // This call passes session data to the Python function "save" in custom.py
  $.ajax({
    "url": "/save",
    "method": "POST",
    "headers": {
      "datatype": "application/json",
      "content-type": "application/json"
    },
    "processData": false,
    "data": filedata
  });
}

/*
function readCount() {
  var count;
  $.ajax({
    url: '/static/text/subject_counter.txt',
    async: false,
    success: function(data){count = data}
  });
  return count.trim();
}

function saveCount(count) {
  $.ajax({
    type:'POST',
    url: '/static/php/save_data.php',
    data: {filename: '/static/text/subject_counter.txt', filedata: count}
  });
}
*/
