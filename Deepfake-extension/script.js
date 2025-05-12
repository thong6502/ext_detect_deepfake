let button;
let modal;
let quality = "720";
let buttonTiktok;
let buttonFacebook;

// Thêm Font Awesome
function loadFontAwesome() {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href =
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css";
  document.head.appendChild(link);
}

function downloadTiktok() {
  var element = document.querySelector('div[class*="DivFlexCenterRow"]');
  if (element) {
    buttonTiktok = document.createElement("button");
    buttonTiktok.innerHTML =
      "<span style='display: flex; align-items: center; justify-content: center;'><i class='fas fa-shield-alt logo-icon' style='margin-right: 6px;'></i>DeepScan</span>";
    buttonTiktok.style.fontWeight = "bold";
    buttonTiktok.style.marginRight = "10px";
    buttonTiktok.style.borderRadius = "24px";
    buttonTiktok.style.padding = "10px 16px";
    buttonTiktok.style.backgroundColor = "#fe2c55";
    buttonTiktok.style.color = "white";
    buttonTiktok.style.border = "none";
    buttonTiktok.style.cursor = "pointer";
    buttonTiktok.style.transition = "all 0.2s ease";
    buttonTiktok.style.boxShadow = "0 2px 8px rgba(254, 44, 85, 0.4)";
    buttonTiktok.style.fontSize = "14px";
    buttonTiktok.style.display = "flex";
    buttonTiktok.style.alignItems = "center";

    buttonTiktok.onmouseover = function () {
      this.style.transform = "scale(1.05)";
      this.style.boxShadow = "0 4px 12px rgba(254, 44, 85, 0.5)";
    };

    buttonTiktok.onmouseout = function () {
      this.style.transform = "scale(1)";
      this.style.boxShadow = "0 2px 8px rgba(254, 44, 85, 0.4)";
    };

    buttonTiktok.addEventListener("click", function () {
      let arrLink = document.querySelector("video").querySelectorAll("source");
      arrLink.forEach((item, index) => {
        console.log(item.src);
      });
      openModal();
    });

    element.appendChild(buttonTiktok);
  }
}

function findSpanByText(text) {
  const spans = document.querySelectorAll("span");
  for (let span of spans) {
    if (span.textContent.trim() === text) {
      return span; // Trả về phần tử <span> nếu văn bản trùng khớp
    }
  }
  return null; // Nếu không tìm thấy phần tử nào
}

function downloadFacebook() {
  var element =
    findSpanByText("Tổng quan").parentNode.parentNode.parentNode.parentNode;
  if (element) {
    if (buttonFacebook) {
      element.removeChild(buttonFacebook);
    }
    buttonFacebook = document.createElement("button");
    buttonFacebook.innerHTML =
      "<span style='display: flex; align-items: center; justify-content: center;'><i class='fas fa-shield-alt logo-icon' style='margin-right: 6px;'></i>DeepScan</span>";
    buttonFacebook.style.fontWeight = "600";
    buttonFacebook.style.gap = "10px";
    buttonFacebook.style.borderRadius = "24px";
    buttonFacebook.style.padding = "10px 16px";
    buttonFacebook.style.backgroundColor = "#1877f2";
    buttonFacebook.style.color = "white";
    buttonFacebook.style.border = "none";
    buttonFacebook.style.cursor = "pointer";
    buttonFacebook.style.transition = "all 0.2s ease";
    buttonFacebook.style.boxShadow = "0 2px 8px rgba(24, 119, 242, 0.4)";
    buttonFacebook.style.fontSize = "14px";
    buttonFacebook.style.display = "flex";
    buttonFacebook.style.alignItems = "center";
    buttonFacebook.id = "down-video";

    buttonFacebook.onmouseover = function () {
      this.style.transform = "scale(1.05)";
      this.style.boxShadow = "0 4px 12px rgba(24, 119, 242, 0.5)";
      this.style.backgroundColor = "#0e6edf";
    };

    buttonFacebook.onmouseout = function () {
      this.style.transform = "scale(1)";
      this.style.boxShadow = "0 2px 8px rgba(24, 119, 242, 0.4)";
      this.style.backgroundColor = "#1877f2";
    };

    buttonFacebook.addEventListener("click", function () {
      var urlFb = window.location.href;
      console.log(urlFb);
      openModal();
    });
    element.appendChild(buttonFacebook);
  } else {
    if (buttonFacebook) {
      buttonFacebook = null;
    }
  }
}

function coverButton() {
  var element = document.querySelector(
    ".style-scope.ytd-download-button-renderer"
  );
  if (element) {
    var rect = element.getBoundingClientRect();
    if (
      rect.width === 0 &&
      rect.height === 0 &&
      rect.top === 0 &&
      rect.left === 0
    ) {
      if (button) {
        document.body.removeChild(button);
        button = null;
      }
      return;
    }
    if (button) {
      document.body.removeChild(button);
    }

    button = document.createElement("button");
    button.innerHTML =
      "<span style='display: flex; align-items: center; justify-content: center;'><i class='fas fa-shield-alt logo-icon' style='margin-right: 6px;'></i>DeepScan</span>";
    button.style.position = "absolute";
    button.style.top = `${rect.top + window.scrollY}px`;
    button.style.left = `${rect.left + window.scrollX}px`;
    button.style.width = `${rect.width}px`;
    button.style.height = `${rect.height}px`;
    button.style.background = "linear-gradient(135deg, #ff416c, #ff4b2b)";
    button.style.color = "#FFFFFF";
    button.style.border = "none";
    button.style.cursor = "pointer";
    button.style.zIndex = "2";
    button.style.borderRadius = "24px";
    button.style.fontWeight = "600";
    button.style.fontSize = "14px";
    button.style.transition = "all 0.3s ease";
    button.style.boxShadow = "0 4px 8px rgba(255, 65, 108, 0.3)";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";

    button.onmouseover = function () {
      this.style.transform = "scale(1.05)";
      this.style.boxShadow = "0 6px 12px rgba(255, 65, 108, 0.4)";
      this.style.background = "linear-gradient(135deg, #ff416c, #ff5236)";
    };

    button.onmouseout = function () {
      this.style.transform = "scale(1)";
      this.style.boxShadow = "0 4px 8px rgba(255, 65, 108, 0.3)";
      this.style.background = "linear-gradient(135deg, #ff416c, #ff4b2b)";
    };

    // Add event listener to open the modal on button click
    button.addEventListener("click", function () {
      openModal();
    });

    document.body.appendChild(button);
  } else {
    if (button) {
      document.body.removeChild(button);
      button = null;
    }
  }
}

// Function to create and open the modal
async function openModal() {
  // Thêm CSS animation
  const style = document.createElement("style");
  style.innerHTML = `
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    @keyframes scaleIn {
      from { transform: translate(-50%, -50%) scale(0.9); opacity: 0; }
      to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }
    
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); }
    }

    @keyframes shake {
      0%, 100% { transform: translateX(0); }
      10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
      20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    @keyframes gradientAnimation {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    @keyframes loadingBar {
      0% { width: 0%; }
      100% { width: 100%; }
    }

    @keyframes float {
      0% { transform: translateY(0px); }
      50% { transform: translateY(-8px); }
      100% { transform: translateY(0px); }
    }
  `;
  document.head.appendChild(style);

  modal = document.createElement("div");
  modal.style.position = "fixed";
  modal.style.top = "0";
  modal.style.left = "0";
  modal.style.width = "100%";
  modal.style.height = "100%";
  modal.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
  modal.style.zIndex = "9999";
  modal.style.animation = "fadeIn 0.3s ease";
  modal.style.backdropFilter = "blur(3px)";

  // Create modal content
  const modalContent = document.createElement("div");
  modalContent.style.position = "absolute";
  modalContent.style.top = "50%";
  modalContent.style.left = "50%";
  modalContent.style.transform = "translate(-50%, -50%)";
  modalContent.style.backgroundColor = "#FFFFFF";
  modalContent.style.padding = "30px";
  modalContent.style.borderRadius = "16px";
  modalContent.style.boxShadow = "0 10px 30px rgba(0, 0, 0, 0.3)";
  modalContent.style.width = "90%";
  modalContent.style.maxWidth = "500px";
  modalContent.style.animation = "scaleIn 0.3s ease-out";
  modalContent.style.display = "flex";
  modalContent.style.flexDirection = "column";
  modalContent.style.alignItems = "center";
  modalContent.style.justifyContent = "center";

  modal.appendChild(modalContent);
  document.body.appendChild(modal);

  // Cải thiện UI quá trình kiểm tra
  const loaderContainer = document.createElement("div");
  loaderContainer.style.display = "flex";
  loaderContainer.style.flexDirection = "column";
  loaderContainer.style.alignItems = "center";
  loaderContainer.style.justifyContent = "center";
  loaderContainer.style.width = "100%";
  loaderContainer.style.padding = "20px 0";

  // Icon animation
  const scanIcon = document.createElement("div");
  scanIcon.innerHTML = `<i class="fas fa-shield-alt logo-icon" style="font-size: 60px; color: #3498db;"></i>`;
  scanIcon.style.animation = "float 2s ease-in-out infinite";
  scanIcon.style.margin = "10px 0 20px 0";

  // Progress container
  const progressContainer = document.createElement("div");
  progressContainer.style.width = "80%";
  progressContainer.style.height = "10px";
  progressContainer.style.backgroundColor = "#f1f1f1";
  progressContainer.style.borderRadius = "5px";
  progressContainer.style.overflow = "hidden";
  progressContainer.style.margin = "25px 0 10px 0";

  // Progress bar
  const progressBar = document.createElement("div");
  progressBar.style.height = "100%";
  progressBar.style.background =
    "linear-gradient(90deg, #3498db, #2ecc71, #3498db)";
  progressBar.style.backgroundSize = "200% 200%";
  progressBar.style.animation =
    "loadingBar 3s ease-in-out infinite, gradientAnimation 2s ease infinite";
  progressContainer.appendChild(progressBar);

  // Loader text
  const loader = document.createElement("div");
  loader.innerHTML = "Đang kiểm tra video, xin vui lòng đợi...";
  loader.style.fontSize = "16px";
  loader.style.fontWeight = "500";
  loader.style.color = "#333";
  loader.style.textAlign = "center";
  loader.style.margin = "15px 0";

  // Thêm stage text
  const stageText = document.createElement("div");
  stageText.innerHTML = "Đang phân tích nội dung video";
  stageText.style.fontSize = "14px";
  stageText.style.color = "#666";
  stageText.style.textAlign = "center";
  stageText.style.margin = "5px 0 20px 0";

  // Đổi thông báo giai đoạn sau mỗi giây
  const stages = [
    "Đang phân tích nội dung video",
    "Đang kiểm tra các đặc điểm của khuôn mặt",
    "Đang so sánh với mô hình deepfake",
    "Đang xác minh tính xác thực",
    "Đang hoàn thiện kiểm tra",
  ];
  let currentStage = 0;
  const stageInterval = setInterval(() => {
    stageText.innerHTML = stages[currentStage];
    currentStage = (currentStage + 1) % stages.length;
  }, 2000);

  loaderContainer.appendChild(scanIcon);
  loaderContainer.appendChild(loader);
  loaderContainer.appendChild(progressContainer);
  loaderContainer.appendChild(stageText);
  modalContent.appendChild(loaderContainer);

  var url = "https://firefly-genuine-adequately.ngrok-free.app/process";
  var url_video = window.location.href;
  var request_data = {
    url: url_video,
  };

  try {
    let response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request_data),
    });

    // Dừng interval cập nhật giai đoạn
    clearInterval(stageInterval);

    if (!response.ok) {
      throw new Error("Kiểm tra thất bại");
    }

    let data = await response.json();

    // Remove loader after data is loaded
    if (modalContent.contains(loaderContainer)) {
      modalContent.removeChild(loaderContainer);
    }

    // Hiển thị kết quả dựa trên loại phản hồi
    const resultContainer = document.createElement("div");
    resultContainer.style.display = "flex";
    resultContainer.style.flexDirection = "column";
    resultContainer.style.alignItems = "center";
    resultContainer.style.justifyContent = "center";
    resultContainer.style.width = "90%";
    resultContainer.style.padding = "20px";
    resultContainer.style.borderRadius = "10px";
    resultContainer.style.margin = "10px 0";
    resultContainer.style.animation = "fadeIn 0.5s ease-out";

    let resultIcon = "";
    let resultTitle = "";
    let resultMessage = "";
    let bgColor = "";
    let textColor = "";
    let borderColor = "";
    let buttonColor = "";
    let buttonHoverColor = "";
    let iconAnimation = "";

    // Xác định kiểu UI dựa trên kết quả từ server
    if (data.result && data.result.includes("REAL")) {
      // Video thật
      resultIcon = "✓";
      resultTitle = "Video Thật";
      resultMessage =
        "Video này được xác nhận là video thật, không có dấu hiệu deepfake.";
      bgColor = "rgba(46, 204, 113, 0.1)";
      textColor = "#27ae60";
      borderColor = "#2ecc71";
      buttonColor = "#2ecc71";
      buttonHoverColor = "#27ae60";
      iconAnimation = "pulse 1.5s infinite";
    } else if (data.result && data.result.includes("FAKE")) {
      // Video giả mạo
      resultIcon = "⚠️";
      resultTitle = "Video Giả Mạo";
      resultMessage =
        "Cảnh báo! Video này có dấu hiệu của deepfake. Hãy cẩn thận khi xem và chia sẻ!";
      bgColor = "rgba(231, 76, 60, 0.1)";
      textColor = "#c0392b";
      borderColor = "#e74c3c";
      buttonColor = "#e74c3c";
      buttonHoverColor = "#c0392b";
      iconAnimation = "shake 0.8s cubic-bezier(.36,.07,.19,.97) both";
    } else if (data.result && data.result.includes("CẢNH BÁO")) {
      // Video cảnh báo
      resultIcon = "ℹ️";
      resultTitle = "Cảnh Báo";
      resultMessage =
        "Video này có một số dấu hiệu đáng ngờ, nhưng không thể kết luận chắc chắn. Hãy xem xét thận trọng.";
      bgColor = "rgba(241, 196, 15, 0.1)";
      textColor = "#d35400";
      borderColor = "#f39c12";
      buttonColor = "#f39c12";
      buttonHoverColor = "#d35400";
      iconAnimation = "pulse 2s infinite";
    } else {
      // Trường hợp không xác định
      resultIcon = "❓";
      resultTitle = "Không Xác Định";
      resultMessage =
        "Không thể xác định tính xác thực của video này. Vui lòng tham khảo thêm các nguồn tin cậy khác.";
      bgColor = "rgba(52, 152, 219, 0.1)";
      textColor = "#2980b9";
      borderColor = "#3498db";
      buttonColor = "#3498db";
      buttonHoverColor = "#2980b9";
    }

    resultContainer.style.backgroundColor = bgColor;
    resultContainer.style.border = `2px solid ${borderColor}`;

    // Icon kết quả
    const iconElement = document.createElement("div");
    iconElement.innerHTML = resultIcon;
    iconElement.style.fontSize = "60px";
    iconElement.style.margin = "10px 0";
    iconElement.style.color = textColor;
    if (iconAnimation) {
      iconElement.style.animation = iconAnimation;
    }

    // Tiêu đề kết quả
    const titleElement = document.createElement("div");
    titleElement.innerHTML = resultTitle;
    titleElement.style.fontSize = "26px";
    titleElement.style.fontWeight = "bold";
    titleElement.style.color = textColor;
    titleElement.style.margin = "10px 0";

    // Thông báo chi tiết
    const messageElement = document.createElement("div");
    messageElement.innerHTML = resultMessage;
    messageElement.style.fontSize = "16px";
    messageElement.style.color = "#333";
    messageElement.style.textAlign = "center";
    messageElement.style.margin = "10px 0 20px 0";
    messageElement.style.lineHeight = "1.5";

    // Nút xem chi tiết
    const linkButton = document.createElement("a");
    linkButton.href = data.output_path;
    linkButton.target = "_blank";
    linkButton.innerHTML = "Xem chi tiết phân tích";
    linkButton.style.display = "inline-block";
    linkButton.style.padding = "12px 24px";
    linkButton.style.backgroundColor = buttonColor;
    linkButton.style.color = "white";
    linkButton.style.borderRadius = "30px";
    linkButton.style.textDecoration = "none";
    linkButton.style.fontWeight = "bold";
    linkButton.style.fontSize = "16px";
    linkButton.style.margin = "15px 0";
    linkButton.style.transition = "all 0.3s ease";
    linkButton.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.1)";

    linkButton.onmouseover = function () {
      this.style.backgroundColor = buttonHoverColor;
      this.style.transform = "translateY(-2px)";
      this.style.boxShadow = "0 6px 8px rgba(0, 0, 0, 0.15)";
    };

    linkButton.onmouseout = function () {
      this.style.backgroundColor = buttonColor;
      this.style.transform = "translateY(0)";
      this.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.1)";
    };

    // Thêm các phần tử vào container
    resultContainer.appendChild(iconElement);
    resultContainer.appendChild(titleElement);
    resultContainer.appendChild(messageElement);
    resultContainer.appendChild(linkButton);

    // Thêm container vào modal
    modalContent.appendChild(resultContainer);
  } catch (error) {
    console.error("Lỗi:", error);

    // Dừng interval cập nhật giai đoạn
    clearInterval(stageInterval);

    // Remove loader after data is loaded or failed
    if (modalContent.contains(loaderContainer)) {
      modalContent.removeChild(loaderContainer);
    }

    // Hiển thị lỗi
    const errorContainer = document.createElement("div");
    errorContainer.style.display = "flex";
    errorContainer.style.flexDirection = "column";
    errorContainer.style.alignItems = "center";
    errorContainer.style.justifyContent = "center";
    errorContainer.style.width = "90%";
    errorContainer.style.padding = "20px";
    errorContainer.style.borderRadius = "10px";
    errorContainer.style.backgroundColor = "rgba(231, 76, 60, 0.1)";
    errorContainer.style.border = "2px solid #e74c3c";
    errorContainer.style.margin = "10px 0";
    errorContainer.style.animation = "fadeIn 0.5s ease-out";

    const errorIcon = document.createElement("div");
    errorIcon.innerHTML = "❌";
    errorIcon.style.fontSize = "48px";
    errorIcon.style.margin = "10px 0";
    errorIcon.style.color = "#e74c3c";

    const errorTitle = document.createElement("div");
    errorTitle.innerHTML = "Không thể kiểm tra";
    errorTitle.style.fontSize = "24px";
    errorTitle.style.fontWeight = "bold";
    errorTitle.style.color = "#e74c3c";
    errorTitle.style.margin = "10px 0";

    const errorMessage = document.createElement("div");
    errorMessage.innerHTML =
      "Đã xảy ra lỗi khi kiểm tra video này. Vui lòng thử lại sau.";
    errorMessage.style.fontSize = "16px";
    errorMessage.style.color = "#333";
    errorMessage.style.textAlign = "center";
    errorMessage.style.margin = "10px 0";
    errorMessage.style.lineHeight = "1.5";

    const retryButton = document.createElement("button");
    retryButton.innerHTML = "Thử lại";
    retryButton.style.padding = "12px 24px";
    retryButton.style.backgroundColor = "#e74c3c";
    retryButton.style.color = "white";
    retryButton.style.border = "none";
    retryButton.style.borderRadius = "30px";
    retryButton.style.fontWeight = "bold";
    retryButton.style.fontSize = "16px";
    retryButton.style.margin = "20px 0";
    retryButton.style.cursor = "pointer";
    retryButton.style.transition = "all 0.3s ease";
    retryButton.style.boxShadow = "0 4px 6px rgba(231, 76, 60, 0.3)";

    retryButton.onmouseover = function () {
      this.style.backgroundColor = "#c0392b";
      this.style.transform = "translateY(-2px)";
      this.style.boxShadow = "0 6px 8px rgba(231, 76, 60, 0.4)";
    };

    retryButton.onmouseout = function () {
      this.style.backgroundColor = "#e74c3c";
      this.style.transform = "translateY(0)";
      this.style.boxShadow = "0 4px 6px rgba(231, 76, 60, 0.3)";
    };

    retryButton.addEventListener("click", function () {
      document.body.removeChild(modal);
      setTimeout(openModal, 300);
    });

    errorContainer.appendChild(errorIcon);
    errorContainer.appendChild(errorTitle);
    errorContainer.appendChild(errorMessage);
    errorContainer.appendChild(retryButton);
    modalContent.appendChild(errorContainer);
  }

  // Cải thiện nút đóng
  const closeButton = document.createElement("button");
  closeButton.innerHTML = "×";
  closeButton.style.position = "absolute";
  closeButton.style.top = "10px";
  closeButton.style.right = "10px";
  closeButton.style.width = "30px";
  closeButton.style.height = "30px";
  closeButton.style.backgroundColor = "#f1f1f1";
  closeButton.style.border = "none";
  closeButton.style.borderRadius = "50%";
  closeButton.style.fontSize = "20px";
  closeButton.style.fontWeight = "bold";
  closeButton.style.color = "#555";
  closeButton.style.cursor = "pointer";
  closeButton.style.display = "flex";
  closeButton.style.alignItems = "center";
  closeButton.style.justifyContent = "center";
  closeButton.style.transition = "all 0.2s";

  closeButton.onmouseover = function () {
    this.style.backgroundColor = "#ddd";
    this.style.transform = "rotate(90deg)";
  };

  closeButton.onmouseout = function () {
    this.style.backgroundColor = "#f1f1f1";
    this.style.transform = "rotate(0deg)";
  };

  closeButton.addEventListener("click", function () {
    modal.style.opacity = "0";
    modal.style.transition = "opacity 0.3s";
    setTimeout(() => {
      document.body.removeChild(modal);
      modal = null;
    }, 300);
  });

  modalContent.appendChild(closeButton);
}

function runExtension() {
  var url = window.location.href;
  if (url.includes("facebook.com")) {
    downloadFacebook();
  } else if (url.includes("youtube.com")) {
    coverButton();
  } else if (url.includes("tiktok.com")) {
    downloadTiktok();
  }
}

// Gọi hàm khi trang tải
window.addEventListener("load", function () {
  loadFontAwesome();
  setInterval(runExtension, 1000);
});
