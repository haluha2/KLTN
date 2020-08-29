function classify() {
  var data = {"text": document.getElementById("tweetText").value}

  $.post("/classify", data, function(data, status){
    var img = document.getElementById('sentimentImg')
    if (data.probability <= 0.65) {
      img.setAttribute('src', 'static/img/neutral.svg');
    }
    else if (data.sentiment == 'Positive') {
      img.setAttribute('src', 'static/img/happy.svg');
    } else {
      img.setAttribute('src', 'static/img/sad.svg');
    }
    document.getElementById('probText').textContent = '(' + data.sentiment + ' with probability: ' + data.probability * 100 + '%)'
  }, "json");
}

function imgPreview(fileInput) {
    var reader = new FileReader();
    reader.onload = function(e) {
      $('#imgOrigin').css('background-image', 'url("' + e.target.result + '")');
    }
    reader.readAsDataURL(fileInput);
}

function Retrieval()
{
    $("#result_imgs").html("");
    $("#count_results").hide();
    var fileUpload = $("#fileInputImg").get(0);
    var files = fileUpload.files;
    var data = new FormData();
    // data.append("function", "loadIMG");
    if (files.length == 1) {
        for (var i = 0; i < files.length; i++) {
            data.append("files", files[i]);
        }

        $.ajax({
            type: "POST",
            url: "/upload",
            contentType: false,
            processData: false,
            data: data,
            beforeSend: function () {
                setLoading(true);
            },
            success: function (path) {
                var ggD_url = "https://drive.google.com/uc?export=view&id=";
                console.log(path);
                var count_rs = path.length;
                imgs_html = "";
                for (i = 0; i < path.length; i++) {
                    imgs_html += "<img class='lazy' width='300' height='150' data-src='" + path[i].id + "' />";
                }
                $("#count_results").text("Found " + count_rs + " results.");
                $("#count_results").show();
                $("#result_imgs").css("height","70vh");
                $("#result_imgs").css("overflow-y","scroll");
                $("#result_imgs").append(imgs_html);
                lazyload(this);
                setLoading(false);
            },
            error: function (err) {
                console.log(err);
                setLoading(false);
            }
        });
    }
}

$("#fileInputImg").on('change', function () {
            var fileUpload = $(this).get(0);
            var files = fileUpload.files;
            var data = new FormData();
            // data.append("function", "loadIMG");
            if (files.length == 1) {
                for (var i = 0; i < files.length; i++) {
                    data.append("files", files[i]);
                    $('#txtImgName').text(files[i].name);
                }

                imgPreview(files[0])
            }
            Retrieval();
});

$("#btnSubmit").on('click', function () {
    $("#result_imgs").html("");
    var fileUpload = $("#fileInputImg").get(0);
    var files = fileUpload.files;
    var data = new FormData();
    // data.append("function", "loadIMG");
    if (files.length == 1) {
        for (var i = 0; i < files.length; i++) {
            data.append("files", files[i]);
        }

        $.ajax({
            type: "POST",
            url: "/upload",
            contentType: false,
            processData: false,
            data: data,
            beforeSend: function () {
                setLoading(true);
            },
            success: function (path) {
                var ggD_url = "https://drive.google.com/uc?export=view&id=";
                console.log(path);
                imgs_html = "";
                for (i = 0; i < path.length; i++) {
                    imgs_html += "<img class='lazy' width='300' height='150' data-src='" + path[i].id + "' />";
                }
                $("#result_imgs").append(imgs_html);
                lazyload(this);
                setLoading(false);

            },
            error: function (err) {
                console.log(err);
                setLoading(false);
            }
        });
    }
});

function setLoading(isLoading) {
    if (isLoading) {
        $('#preloader').show();
        //$("#loader").show();
        //$('body').css({ 'opacity': 0.5 });
    }
    else {
        $('#preloader').hide();
        //$("#loader").hide();
        //$('body').css({ 'opacity': 1 });
    }
};

// Prevent default submit behaviour
$("#tweet_form").submit(function(e) {
    e.preventDefault();
});

function lazyload(window) {
    var ggD_url = "https://drive.google.com/uc?export=view&id=";
    var $q = function (q, res) {
        if (document.querySelectorAll) {
            res = document.querySelectorAll(q);
        } else {
            var d = document
                , a = d.styleSheets[0] || d.createStyleSheet();
            a.addRule(q, 'f:b');
            for (var l = d.all, b = 0, c = [], f = l.length; b < f; b++)
                l[b].currentStyle.f && c.push(l[b]);

            a.removeRule(0);
            res = c;
        }
        return res;
    }
        , addEventListener = function (evt, fn) {
            window.addEventListener
                ? this.addEventListener(evt, fn, false)
                : (window.attachEvent)
                    ? this.attachEvent('on' + evt, fn)
                    : this['on' + evt] = fn;
        }
        , _has = function (obj, key) {
            return Object.prototype.hasOwnProperty.call(obj, key);
        }
        ;

    function loadImage(el, fn) {
        var img = new Image()
            , src = ggD_url+el.getAttribute('data-src');
        img.onload = function () {
            if (!!el.parent)
                el.parent.replaceChild(img, el)
            else
                el.src = src;

            fn ? fn() : null;
        }
        img.src = src;
    }

    function elementInViewport(el) {
        var rect = el.getBoundingClientRect()

        return (
            rect.top >= 0
            && rect.left >= 0
            && rect.top <= (window.innerHeight || document.documentElement.clientHeight)
        )
    }

    var images = new Array()
        , query = $q('img.lazy')
        , processScroll = function () {
            for (var i = 0; i < images.length; i++) {
                if (elementInViewport(images[i])) {
                    loadImage(images[i], function () {
                        images.splice(i, i);
                    });
                }
            };
        }
        ;
    // Array.prototype.slice.call is not callable under our lovely IE8 
    for (var i = 0; i < query.length; i++) {
        images.push(query[i]);
    };

    processScroll();
    //addEventListener('scroll', processScroll);
    $("#result_imgs").scroll(processScroll);

};
/*
$(function() {
    //Inset Dark
    $("#rs").mCustomScrollbar({
      theme: "inset-3-dark"
    });
});
*/