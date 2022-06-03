// 리뷰 별점 작동여부 체크 함수
function reviewCheck() {
    var reviewForm = document.reviewForm;
    if(reviewForm.appearance.value == 0 || reviewForm.aroma.value == 0 || reviewForm.palate.value == 0 || reviewForm.taste.value == 0){
        alert("별점을 모두 입력해주세요!")
    }
    else{
        reviewForm.submit();
    }

}