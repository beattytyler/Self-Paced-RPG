document.addEventListener("DOMContentLoaded", function () {
  // Get all topic buttons
  const topicButtons = document.querySelectorAll(".button[data-topic]");
  const closeButton = document.querySelector(".close-button");
  const videoModal = document.getElementById("video-modal");
  const videoIframe = document.getElementById("video-iframe");
  const videoTitle = document.getElementById("current-video-title");

  // YouTube API variables
  let ytPlayer = null;
  let isAdminOverride = false;
  let currentVideoId = null;
  let currentTopic = null;

  // Load progress from server when page loads
  loadProgress();

  // Check if user is admin
  checkAdminStatus();

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

  // Check if user has admin privileges
  function checkAdminStatus() {
    fetch("/api/admin/status")
      .then((response) => response.json())
      .then((data) => {
        isAdminOverride = data.is_admin || false;
        if (isAdminOverride) {
          console.log("Admin override enabled - video completion not required");
        }
      })
      .catch((error) => {
        console.log("Admin status check failed, assuming non-admin");
        isAdminOverride = false;
      });
  }

  // YouTube API ready callback
  function onYouTubeIframeAPIReady() {
    console.log("YouTube API Ready");
  }

  // Initialize YouTube player with enhanced tracking
  function initializePlayer(videoId, elementId) {
    if (ytPlayer) {
      ytPlayer.destroy();
    }

    ytPlayer = new YT.Player(elementId, {
      height: "400",
      width: "100%",
      videoId: videoId,
      playerVars: {
        autoplay: 1,
        controls: 1,
        rel: 0,
        showinfo: 0,
        modestbranding: 1,
      },
      events: {
        onReady: onPlayerReady,
        onStateChange: onPlayerStateChange,
      },
    });
  }

  function onPlayerReady(event) {
    console.log("YouTube player ready");
    startVideoProgressTracking();
  }

  function onPlayerStateChange(event) {
    if (event.data == YT.PlayerState.ENDED) {
      handleVideoComplete();
    }
  }

  // Start enhanced video progress tracking
  function startVideoProgressTracking() {
    if (!ytPlayer || !currentTopic) return;

    const trackingInterval = setInterval(() => {
      if (!ytPlayer || ytPlayer.getPlayerState() === YT.PlayerState.UNSTARTED) {
        clearInterval(trackingInterval);
        return;
      }

      const currentTime = ytPlayer.getCurrentTime();
      const duration = ytPlayer.getDuration();

      if (duration > 0) {
        const progress = Math.min((currentTime / duration) * 100, 100);
        updateVideoProgress(currentTopic, progress);

        // Auto-complete for admins at 80% or if override enabled
        if (isAdminOverride && progress >= 80) {
          handleVideoComplete();
          clearInterval(trackingInterval);
        }
      }
    }, 2000); // Check every 2 seconds

    // Store interval reference for cleanup
    videoModal.dataset.trackingInterval = trackingInterval;
  }

  function handleVideoComplete() {
    if (currentTopic) {
      saveProgress(currentTopic, 100);

      // Redirect to quiz for functions topic
      if (currentTopic === "functions") {
        setTimeout(() => {
          window.location.href = "/quiz/functions";
        }, 1000);
      }
    }
  }

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
    currentTopic = topic;

    // Determine the correct API endpoint based on page context
    let apiUrl;
    const currentPath = window.location.pathname;

    if (currentPath.startsWith("/subjects/")) {
      // We're on a subject page, extract subject from URL
      const pathParts = currentPath.split("/");
      const subject = pathParts[2]; // /subjects/{subject}
      const subtopic = topic; // The topic is actually the subtopic ID

      // For subject pages, we need to get the first video from the subtopic
      // The API expects /api/video/{subject}/{subtopic}/{videoKey}
      // We'll fetch the video data to get the available video keys
      apiUrl = `/api/video/${subject}/${subtopic}/${subtopic}`; // Assuming video key matches subtopic
    } else {
      // Legacy behavior for results page
      apiUrl = `/api/video/${topic}`;
    }

    // Get video data from API
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => {
        // Set video title
        videoTitle.textContent = data.title;

        // Extract video ID from YouTube URL
        const videoId = extractVideoId(data.url);
        currentVideoId = videoId;

        if (videoId) {
          // Create YouTube player container if it doesn't exist
          if (!document.getElementById("youtube-player")) {
            videoIframe.outerHTML = '<div id="youtube-player"></div>';
          }

          // Initialize YouTube player with enhanced tracking
          setTimeout(() => {
            initializePlayer(videoId, "youtube-player");
          }, 100);
        } else {
          // Fallback to iframe for non-YouTube videos
          videoIframe.src = data.url;

          // Legacy handling for functions topic
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
        }

        // Show modal
        videoModal.style.display = "flex";
        videoModal.dataset.currentTopic = topic;

        // Add admin override button if user is admin
        if (isAdminOverride) {
          addAdminOverrideButton();
        }
      })
      .catch((error) => console.error("Error loading video data:", error));
  }

  // Extract YouTube video ID from various URL formats
  function extractVideoId(url) {
    const regExp =
      /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  }

  // Add admin override button to video modal
  function addAdminOverrideButton() {
    if (document.getElementById("admin-override-btn")) return;

    const overrideBtn = document.createElement("button");
    overrideBtn.id = "admin-override-btn";
    overrideBtn.textContent = "Admin: Mark Complete";
    overrideBtn.className = "admin-override-button";
    overrideBtn.style.cssText = `
      position: absolute;
      top: 10px;
      right: 50px;
      background: #dc3545;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      z-index: 1001;
    `;

    overrideBtn.addEventListener("click", function () {
      handleVideoComplete();
      closeVideo();
    });

    videoModal.querySelector(".modal-content").appendChild(overrideBtn);
  }

  // Close video modal
  function closeVideo() {
    // Get current topic from modal attribute
    const topic = videoModal.dataset.currentTopic;
    console.log("Closing video modal for topic:", topic); // Debug log

    // Stop YouTube player if it exists
    if (ytPlayer) {
      ytPlayer.stopVideo();
    }

    // Clear tracking intervals
    const trackingInterval = videoModal.dataset.trackingInterval;
    if (trackingInterval) {
      clearInterval(trackingInterval);
    }

    // Remove admin override button
    const overrideBtn = document.getElementById("admin-override-btn");
    if (overrideBtn) {
      overrideBtn.remove();
    }

    // Stop video playback
    videoIframe.src = "";
    videoModal.style.display = "none";
    stopTracking();

    // Reset current tracking variables
    currentTopic = null;
    currentVideoId = null;

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

  // Update video progress during playback
  function updateVideoProgress(topic, progress) {
    const videoProgressBar = document.getElementById("video-progress-bar");
    if (videoProgressBar) {
      videoProgressBar.style.width = `${progress}%`;
    }

    // Save progress every 10% increment to avoid too many API calls
    if (Math.floor(progress) % 10 === 0) {
      saveProgress(topic, progress);
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
