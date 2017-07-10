function show_nav_bar() {
    var x = document.getElementById('navlist');
    if (x.className === 'menue') {
      x.className = 'menue_show';
    } else {
      x.className = 'menue';
    }
}
