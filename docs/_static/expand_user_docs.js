// Expand the User Documentation section (first sidebar caption) by default.
// Marks it "current", as the theme does when the section is clicked open, so
// expand/collapse keeps working. Runs on "load", after the theme's own reset.
window.addEventListener("load", function () {
    var item = document.querySelector(
        ".wy-menu-vertical p.caption:first-of-type + ul > li.toctree-l1"
    );
    if (item && item.querySelector("ul")) {
        item.classList.add("current");
        item.setAttribute("aria-expanded", "true");
    }
});
