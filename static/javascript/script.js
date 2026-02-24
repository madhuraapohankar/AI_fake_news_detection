// ================= THEME TOGGLE =================
document.addEventListener("DOMContentLoaded", function () {
    const toggleBtn = document.getElementById("themeToggle");

    if (toggleBtn) {
        toggleBtn.addEventListener("click", function () {
            document.body.classList.toggle("light-mode");

            // optional: change icon
            if (document.body.classList.contains("light-mode")) {
                toggleBtn.textContent = "☀️";
            } else {
                toggleBtn.textContent = "🌙";
            }
        });
    }
});