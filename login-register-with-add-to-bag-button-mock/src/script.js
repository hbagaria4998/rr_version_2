$(function() {
  $(".add-to-bag-button").click(function() {
    $(this).toggleClass("open");
    $("#reg-login-details-box").toggleClass("open");
    $('.reg-login-item').each(function(i) {
      var $item = $(this);
      setTimeout(function() {
        $item.toggleClass('heyYoBro');
      }, i*10);
    });
  });
});