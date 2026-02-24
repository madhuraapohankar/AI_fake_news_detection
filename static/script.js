document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ script loaded");

    const btn = document.getElementById("themeToggle");

    if (!btn) {
        console.error("❌ themeToggle button NOT found");
        return;
    }

    // Load saved theme
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme === "light") {
        document.body.classList.add("light-mode");
        btn.textContent = "☀️";
    } else {
        btn.textContent = "🌙";
    }

    // Toggle click
    btn.addEventListener("click", function () {
        console.log("✅ toggle clicked");

        document.body.classList.toggle("light-mode");

        if (document.body.classList.contains("light-mode")) {
            btn.textContent = "☀️";
            localStorage.setItem("theme", "light");
        } else {
            btn.textContent = "🌙";
            localStorage.setItem("theme", "dark");
        }
    });
});