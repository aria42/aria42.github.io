$(".full img").on("click", function() {
  $(this).toggleClass("zoom");
});

SVGElement.prototype.addClass = function (className) {
  if (!this.hasClass(className)) {
    this.setAttribute('class', this.getAttribute('class') + ' ' + className);
  }
};

$("a.nav-text-button").on({ 'touchstart' : function(){
  $(this).addClass('active');
}});

$("a.nav-text-button").on({ 'touchend' : function(){
  $(this).removeClass('active');
}});
$("a.nav-text-button").on({ 'touchleave' : function(){
  $(this).removeClass('active');
}});
