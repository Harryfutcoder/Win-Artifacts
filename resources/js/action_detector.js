const selectors = arguments[0];
const textDetectModeInput = arguments[1];
const textDetectMode = Object.freeze({
  INNER_TEXT: 0,
  LABEL: 1
});
const elements = selectors
    .flatMap(selector => Array.from(document.querySelectorAll(selector)))
return elements.map(elem => ({
    xpath: generateXpath(elem),
    visible: isVisible(elem),
    text: getText(elem)
}));

function isVisible(elem) {
    if (!(elem instanceof Element)) {
        return false;
    }
    const style = getComputedStyle(elem);
    if (style.display === 'none' || style.visibility !== 'visible' || style.opacity < 0.1) {
        return false;
    }
    if (elem.offsetWidth + elem.offsetHeight + elem.getBoundingClientRect().height + elem.getBoundingClientRect().width === 0) {
        return false;
    }
    const elemCenter = {
        x: document.documentElement.scrollLeft + elem.getBoundingClientRect().left + elem.offsetWidth / 2,
        y: document.documentElement.scrollTop + elem.getBoundingClientRect().top + elem.offsetHeight / 2
    };
    if (elemCenter.x < 0) return false;
    if (elemCenter.x > document.documentElement.offsetWidth) return false;
    if (elemCenter.y < 0) return false;
    if (elemCenter.y > document.documentElement.offsetHeight) return false;
    let pointContainer = elementFromAbsolutePoint(elemCenter.x, elemCenter.y);
    while (pointContainer) {
        if (pointContainer === elem) {
            return true;
        }
        pointContainer = pointContainer.parentNode;
    }
    return false;
}

function elementFromAbsolutePoint(x, y) {
    const scrollX = document.documentElement.scrollLeft;
    const scrollY = document.documentElement.scrollTop;
    window.scrollTo(x - window.innerWidth / 2, y - window.innerHeight / 2);
    const newX = x - document.documentElement.scrollLeft;
    const newY = y - document.documentElement.scrollTop;
    const elem = document.elementFromPoint(newX, newY);
    window.scrollTo(scrollX, scrollY);
    return elem;
}

function generateXpath(element) {
    let tag = element.tagName.toLowerCase();
//    if (element.id) {
//        return `//${tag}[@id="${element.id}"]`
//    } else
    if (element === document.documentElement) {
        return '/html'
    } else {
        const sameTagSiblings = Array.from(element.parentNode.childNodes)
            .filter(e => e.nodeName === element.nodeName)
        const idx = sameTagSiblings.indexOf(element)

        return `${generateXpath(element.parentNode)}/${tag}${sameTagSiblings.length > 1 ? `[${idx + 1}]` : ''}`
    }
}

function getText(element) {
    if (textDetectModeInput === textDetectMode.INNER_TEXT) {
        return element.innerText
    }
    else if (textDetectModeInput === textDetectMode.LABEL) {
        const label = document.querySelector(`label[for="${element.id}"]`)
        if (label) {
            return label.innerText;
        }
        else {
            return "";
        }
    }
    else {
        return "";
    }
}
