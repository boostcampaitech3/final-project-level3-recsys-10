window.onload = function(){
    appearance = document.querySelectorAll('.appearance-star span')
    appearanceValue = document.querySelectorAll('.appearance-star input')
    for (var i = 0; i < appearance.length; i++) {
        appearance[i].style.width = String(appearanceValue[i].defaultValue*10) + '%';
    }

    aroma = document.querySelectorAll('.aroma-star span')
    aromaValue = document.querySelectorAll('.aroma-star input')
    for (var i = 0; i < aroma.length; i++) {
        aroma[i].style.width = String(aromaValue[i].defaultValue*10) + '%';
    }

    palate = document.querySelectorAll('.palate-star span')
    palateValue = document.querySelectorAll('.palate-star input')
    for (var i = 0; i < palate.length; i++) {
        palate[i].style.width = String(palateValue[i].defaultValue*10) + '%';
    }

    taste = document.querySelectorAll('.taste-star span')
    tasteValue = document.querySelectorAll('.taste-star input')
    for (var i = 0; i < taste.length; i++) {
        taste[i].style.width = String(tasteValue[i].defaultValue*10) + '%';
    }
}