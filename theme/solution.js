const solutionText = document.querySelector("#solution");
const solutionEl = document.querySelector(".solution-details");
const summary = solutionEl.querySelector("summary");
const solutionExplanation = document.querySelector(".solution-explanation");
summary.textContent = "Reveal Solution";

let lottie = document.createElement("lottie-player");
lottie.classList.add("solution-lottie");
lottie.setAttribute("src", "/animations/solution.json");
lottie.setAttribute("background", "transparent");
lottie.setAttribute("speed", "1");
lottie.setAttribute("loop", true);
lottie.setAttribute("autoplay", true);
lottie.style = "width: 240px; height: 212px;";

solutionEl.before(lottie);

// Add event listener for details open/close
solutionEl.addEventListener("toggle", function () {
  solutionEl.classList.add("scroll-margin");
  if (this.open) {
    // When details is opened, scroll to show all content
    summary.textContent = "Hide Solution";
    solutionText.classList.add('scroll-margin');
    solutionText.scrollIntoView({ behavior: "smooth", block: "start" });
    lottie.classList.add("solution-lottie-hidden");
  } else {
    summary.textContent = "Reveal Solution";
    lottie.classList.remove("solution-lottie-hidden");
  }
});
