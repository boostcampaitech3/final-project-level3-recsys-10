const appearanceStar = (target) => {
    document.querySelector('#appearance-star span').style.width = `${target.value * 10}%`;

}

const aromaStar = (target) => {
    document.querySelector('#aroma-star span').style.width = `${target.value * 10}%`;

}

const palateStar = (target) => {
    document.querySelector('#palate-star span').style.width = `${target.value * 10}%`;

}

const tasteStar = (target) => {
    document.querySelector('#taste-star span').style.width = `${target.value * 10}%`;

}