/* ===== DARK / LIGHT TOGGLE ===== */
const toggleBtn = document.getElementById("themeToggle");

toggleBtn.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");

    toggleBtn.textContent =
        document.body.classList.contains("light-mode") ? "☀️" : "🌙";
});

/* ===== PARTICLES BACKGROUND ===== */
particlesJS("particles-js", {
    particles: {
        number: { value: 55 },
        color: { value: "#60a5fa" },
        opacity: { value: 0.35 },
        size: { value: 3 },
        move: { speed: 1.2 }
    }
});
/* ===== SHOW LOADER ===== */
function showLoader() {
    const loader = document.getElementById("loader");
    if (loader) loader.classList.remove("hidden");
}