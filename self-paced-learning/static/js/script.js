document.addEventListener("DOMContentLoaded", function () {
  // Get all topic buttons
  const topicButtons = document.querySelectorAll(".button[data-topic]");
  const closeButton = document.querySelector(".close-button");
  const videoModal = document.getElementById("video-modal");
  const videoIframe = document.getElementById("video-iframe");
  const videoTitle = document.getElementById("current-video-title");

  // Load progress from server hen page loads
  loadProgress();

  // Add click event to all topic buttons
  topicButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const topic = this.getAttribute("data-topic");
      openVideo(topic);
    });
  });

  // Close video modal
  closeButton.addEventListener("click", function () {
    closeVideo();
  });

  // Close modal when clicking outside the video
  videoModal.addEventListener("click", function (e) {
    if (e.target === videoModal) {
      closeVideo();
    }
  });

  // Load progress from server API
  function loadProgress() {
    fetch("/api/progress")
      .then((response) => response.json())
      .then((data) => {
        // Update progress bars based on data
        for (const topic in data) {
          updateProgressBar(topic, data[topic]);
        }
      })
      .catch((error) => console.error("Error loading progress:", error));
  }

  // Open video modal with selected topic
  function openVideo(topic) {
    // Get video data from API
    fetch(`/api/video/${topic}`)
      .then((response) => response.json())
      .then((data) => {
        // Set video title
        videoTitle.textContent = data.title;

        // Set video iframe source
        videoIframe.src = data.url;

        if (topic === "functions") {
          videoIframe.onload = function () {
            videoIframe.contentWindow.postMessage(
              JSON.stringify({ event: "listening" }),
              "*"
            );

            window.addEventListener("message", function onVideoEnd(event) {
              let data;
              try {
                data =
                  typeof event.data === "string"
                    ? JSON.parse(event.data)
                    : event.data;
              } catch (e) {
                return;
              }

              if (data.event === "onStateChange" && data.info === 0) {
                // 0 = ended
                window.removeEventListener("message", onVideoEnd);
                window.location.href = "/quiz/functions";
              }
            });
          };
        }
        // Show modal
        videoModal.style.display = "flex";
        videoModal.dataset.currentTopic = topic;

        // Start tracking video progress
        startTracking(topic);
      })
      .catch((error) => console.error("Error loading video data:", error));
  }

  // Close video modal
  function closeVideo() {
    // Get current topic from modal attribute
    const topic = videoModal.dataset.currentTopic;
    console.log("Closing video modal for topic:", topic); // Debug log

    // Stop video playback
    videoIframe.src = "";
    videoModal.style.display = "none";
    stopTracking();

    // Redirect to quiz page if topic is "functions"
    if (topic === "functions") {
      console.log("Redirecting to /quiz/functions");
      window.location.href = "/quiz/functions";
    }
  }

  // Track video progress (simulation for demonstration)
  let trackingInterval;
  function startTracking(topic) {
    // Get current progress
    fetch("/api/progress")
      .then((response) => response.json())
      .then((data) => {
        let currentProgress = data[topic] || 0;
        const videoProgressBar = document.getElementById("video-progress-bar");

        // Update progress bar initially
        videoProgressBar.style.width = `${currentProgress}%`;

        // Clear any existing interval
        if (trackingInterval) {
          clearInterval(trackingInterval);
        }

        // Update progress every 3 seconds (simulating video playback)
        // In a real implementation, you would use the video's timeupdate event
        trackingInterval = setInterval(() => {
          if (currentProgress < 100) {
            currentProgress += 5; // Increase by 5% each interval
            if (currentProgress > 100) currentProgress = 100;

            // Update progress bar
            videoProgressBar.style.width = `${currentProgress}%`;

            // Save progress to server
            saveProgress(topic, currentProgress);

            // If completed, stop tracking
            if (currentProgress === 100) {
              clearInterval(trackingInterval);
            }
          }
        }, 3000);
      });
  }

  function stopTracking() {
    if (trackingInterval) {
      clearInterval(trackingInterval);
    }
  }

  // Save progress to server
  function saveProgress(topic, progress) {
    fetch("/api/progress/update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic: topic,
        progress: progress,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Update main page progress bar
        updateProgressBar(topic, progress);
      })
      .catch((error) => console.error("Error saving progress:", error));
  }

  // Update topic progress bar and completion badge
  function updateProgressBar(topic, progress) {
    const progressBar = document.getElementById(`${topic}-progress`);
    const completionBadge = document.getElementById(`${topic}-badge`);

    if (progressBar) {
      progressBar.style.width = `${progress}%`;

      // Show completion badge if 100%
      if (progress === 100 && completionBadge) {
        completionBadge.style.display = "inline";
      }
    }
  }
});
