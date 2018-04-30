// function to show or hide navigation bar
function show_nav_bar() {
    var x = document.getElementById('navlist');
    if (x.className === 'menue') {
      x.className = 'menue_show';
    } else {
      x.className = 'menue';
    }
}

// create table of contents automatically
$(document).ready(function(){
  var ToC =
    "<section role='navigation' class='table-of-contents' " +
    "style='width:30%;position:fixed;margin:60px 70%;'>" +
      "<h4>Table of Contents:</h4>" +
      "<ul style='list-style-type:none;margin-left:0;padding-left:0;'>";

  $("main h2, main h3").each(function() {
    var current = $(this)
    var name = current.text();
    current.attr("id", name)
    current.attr('style', "padding-top:60px;margin-top: -60px")
    ToC +=
      "<li>" + "&nbsp;".repeat(current.prop("tagName")[1]*4-6) +
      "<a href='#" + name + "'>" +
        name +
      "</a> </li>";
  });

  ToC +=
   "</ul>" +
  "</section>";

  $("body").prepend(ToC);
});
