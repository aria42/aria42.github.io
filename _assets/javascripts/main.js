$(".full img").on("click", function() {
  $(this).toggleClass("zoom");
});

SVGElement.prototype.addClass = function (className) {
  if (!this.hasClass(className)) {
    this.setAttribute('class', this.getAttribute('class') + ' ' + className);
  }
};

function svgFill() {
  $('img[src$="svg"]').hide()
    .each(function(i, item) {
      var _this = this;
      return $.get(this.src).success(function(data) {
        var $svg, a, nName, nValue, _attr, _i, _len;
        $svg = $(data).find('svg');
        _attr = _this.attributes;
        $.extend(_attr, $svg[0].attributes);
        for (_i = 0, _len = _attr.length; _i < _len; _i++) {
          a = _attr[_i];
          nName = a.nodeName;
          nValue = a.nodeValue;
          if (nName !== 'src' && nName !== 'style') {
            $svg.attr(nName, nValue);
          }
        }
        return $(_this).replaceWith($svg);
      });
  });
}

var accented = "#428bca";

function setupHeader() {
  var uri = window.location.pathname.substring(1);
  if (uri.indexOf("/") >= 0) {
    uri = uri.substring(0, uri.indexOf("/"));
  }
  if (uri == "") {
    var loop = setInterval(function() {
      if ($(".nav-button-home svg ellipse").length) {
        $(".nav-button-home svg ellipse").attr("fill",accented);
      }
    }, 5)
  } else {
    $(".nav-button-" + uri).addClass("accented");
  }
}

(function() {
  $(svgFill);
  $(setupHeader);
}).call(this);
