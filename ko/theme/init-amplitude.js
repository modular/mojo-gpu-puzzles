async function wait(ms = 15) {
  await new Promise((resolve) =>
    setTimeout(
      () =>
        requestAnimationFrame(() => {
          resolve();
        }),
      ms
    )
  );
}

async function waitUntil(condition, maxChecks = 1000) {
  let timesChecked = 0;
  while (!condition() && timesChecked < maxChecks) {
    timesChecked = timesChecked + 1;
    await wait();
  }
}
const is_prod = window.location.hostname.includes("modular.com");
const script = document.createElement("script");
const amplitudeKey = is_prod
  ? "3878a0571d1575870a7d0a5f7e644d23"
  : "d8bf208ebdc1b1000d38da8b826a74c4";
script.src = `https://cdn.amplitude.com/script/${amplitudeKey}.js`;
document.head.appendChild(script);
waitUntil(() => window.amplitude).then(() => {
  window.amplitude.init(amplitudeKey, {
    fetchRemoteConfig: true,
    autocapture: true,
  });
});

let timeStartedOnPage = new Date();
function trackTimeOnPage(pathname) {
  if (!timeStartedOnPage) {
    return;
  }
  const durationInSeconds =
    Math.round(new Date().getTime() - timeStartedOnPage.getTime()) / 1000;
  window.amplitude.track("TimeOnPage", {
    duration: `${Math.round(durationInSeconds)}`,
    minutes: `${Math.round(durationInSeconds / 60)}`,
    pathname,
  });
}
window.addEventListener("beforeunload", () => {
  const { pathname } = window.location;
  trackTimeOnPage(pathname);
  amplitude.track("MaxScrollPercentage", { maxScroll, pathname });
  return undefined;
});
