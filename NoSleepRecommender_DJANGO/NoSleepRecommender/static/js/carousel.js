$(document).ready(function() {
    $("#show").hide();
    $("#hide").click(function(){
        console.log('hide');
        $("#story_body").hide();
        $("#hide").hide();
        $("#show").show();
    });
    $("#show").click(function(){
        $("#story_body").show();
        $("#hide").show();
        $("#show").hide();
    });
  $("#myCarousel").on("slide.bs.carousel", function(e) {
   console.log('inside');
    var $e = $(e.relatedTarget);
    var idx = $e.index();
    var itemsPerSlide = 3;
    var totalItems = $(".carousel-item").length;

    if (idx >= totalItems - (itemsPerSlide - 1)) {
      var it = itemsPerSlide - (totalItems - idx);
      for (var i = 0; i < it; i++) {
        // append slides to end
        if (e.direction == "left") {
          $(".carousel-item")
            .eq(i)
            .appendTo(".carousel-inner");
        } else {
          $(".carousel-item")
            .eq(0)
            .appendTo($(this).find(".carousel-inner"));
        }
      }
    }
  });
});