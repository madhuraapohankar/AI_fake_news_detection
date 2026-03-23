// Prevent Form Resubmission on Page Refresh
if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
}

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
/* ================= PROFILE DROPDOWN ================= */

document.addEventListener("click", function (event) {

    const profileContainer = document.querySelector(".profile-container");
    const profileCard = document.getElementById("profileCard");

    if (!profileContainer || !profileCard) return;

    // If clicking on profile icon → toggle
    if (profileContainer.contains(event.target)) {

        if (event.target.closest(".profile-icon")) {
            profileCard.style.display =
                profileCard.style.display === "block" ? "none" : "block";
        }

    } else {
        // Click outside → close
        profileCard.style.display = "none";
    }
});

/* ================= CLEAR RESULT ON NEW INPUT ================= */
document.addEventListener("DOMContentLoaded", function () {
    const inputs = document.querySelectorAll("textarea, input[type='url'], input[type='text'], input[type='file']");
    const resultBoxes = document.querySelectorAll(".result-box");

    if (inputs.length > 0 && resultBoxes.length > 0) {
        inputs.forEach(input => {
            // Listen for typing or file selection
            input.addEventListener("input", function() {
                resultBoxes.forEach(box => {
                    box.style.transition = "opacity 0.3s ease, transform 0.3s ease";
                    box.style.opacity = "0";
                    box.style.transform = "translateY(10px) translateZ(0px)";
                    setTimeout(() => {
                        box.style.display = "none";
                    }, 300);
                });
            }, { once: true }); // Only trigger once so we aren't spamming styles
        });
    }
});