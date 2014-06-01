$(".full img").on("click", function() {
  $(this).toggleClass("zoom");
});

(function() {

  $(function() {
    var $targets;
    $targets = $('img[src$="svg"]').hide();
    return $targets.each(function(i, item) {
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
  });

}).call(this);
