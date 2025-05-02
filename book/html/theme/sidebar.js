let sidebar = null;
const body = document.body;
const sidebar_toggle = document.getElementById("sidebar-toggle-anchor");
const isDesktop = window.innerWidth >= 1220;
sidebar_toggle.checked = isDesktop;
body.classList.add("sidebar-" + (isDesktop ? "visible" : "hidden"));

function setAriaExpanded() {
  document.getElementById("sidebar-toggle").setAttribute("aria-expanded", sidebar === "visible");
}

function resetSidebarClasses() {
  body.classList.remove("sidebar-visible");
  body.classList.remove("sidebar-hidden");
  body.classList.add("sidebar-" + sidebar);
}

window.addEventListener("resize", function () {
  if (window.innerWidth >= 1220) {
    try {
      sidebar = localStorage.getItem("mdbook-sidebar");
    } catch (e) {}
    sidebar_toggle.checked = true;
    sidebar = "visible";
  } else {
    sidebar_toggle.checked = false;
    sidebar = "hidden";
  }
  sidebar_toggle.checked = sidebar === "visible";
  resetSidebarClasses();
  setAriaExpanded();
});

const collapseSidebar = document.querySelector(".collapse-sidebar");
collapseSidebar.addEventListener("click", function (event) {
  event.preventDefault();
  sidebar_toggle.checked = false;
  sidebar = "hidden";
  resetSidebarClasses();
  setAriaExpanded();
});

setAriaExpanded();
Array.from(document.querySelectorAll("#sidebar a")).forEach(function (link) {
  link.setAttribute("tabIndex", sidebar === "visible" ? 0 : -1);
});
