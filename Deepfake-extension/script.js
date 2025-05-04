let button;
let modal;
let quality = "720";
let buttonTiktok;
let buttonFacebook;

function downloadTiktok() {
  var element = document.querySelector('div[class*="DivFlexCenterRow"]');
  if (element) {
    buttonTiktok = document.createElement("button");
    buttonTiktok.innerHTML = "Check video";
    buttonTiktok.style.fontWeight = "bold";
    buttonTiktok.style.marginRight = "10px";
    buttonTiktok.style.borderRadius = "10px";

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
    buttonFacebook.innerHTML = "Check video";
    buttonFacebook.style.fontWeight = "bold";
    buttonFacebook.style.gap = "10px";
    buttonFacebook.style.borderRadius = "10px";
    //buttonFacebook.style.height = "30px"
    buttonFacebook.id = "down-video";

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
    button.innerHTML = "Check video";
    button.style.position = "absolute";
    button.style.top = `${rect.top + window.scrollY}px`;
    button.style.left = `${rect.left + window.scrollX}px`;
    button.style.width = `${rect.width}px`;
    button.style.height = `${rect.height}px`;
    button.style.backgroundColor = "#ee0979";
    button.style.background = "linear-gradient(to right, #ff6a00, #ee0979)";
    button.style.color = "#FFFFFF";
    button.style.border = "none";
    button.style.cursor = "pointer";
    button.style.zIndex = "2";
    button.style.borderRadius = "20px";
    button.style.fontWeight = "bold";

    // Add event listener to open the modal on button click
    button.addEventListener("click", function () {
      openModal();
    });

    document.body.appendChild(button);
    // console.log("Button added at the element's position.");
  } else {
    if (button) {
      document.body.removeChild(button);
      button = null;
    }
    // console.log("Element not found!");
  }
}

// Function to create and open the modal
async function openModal() {
  modal = document.createElement("div");
  modal.style.position = "fixed";
  modal.style.top = "0";
  modal.style.left = "0";
  modal.style.width = "100%";
  modal.style.height = "100%";
  modal.style.backgroundColor = "rgba(0, 0, 0, 0.7)"; // Semi-transparent background
  modal.style.zIndex = "9999"; // Ensure it is on top

  // Create modal content
  const modalContent = document.createElement("div");
  modalContent.style.position = "absolute";
  modalContent.style.top = "50%";
  modalContent.style.left = "50%";
  modalContent.style.transform = "translate(-50%, -50%)";
  modalContent.style.backgroundColor = "#FFFFFF";
  modalContent.style.padding = "20px";
  modalContent.style.borderRadius = "10px";
  modalContent.style.boxShadow = "0 0 10px rgba(0, 0, 0, 0.5)";
  modalContent.style.height = "50%";
  modalContent.style.width = "50%";

  // Đặt modalContent thành flexbox để căn giữa nội dung bên trong
  modalContent.style.display = "flex";
  modalContent.style.flexDirection = "column";
  modalContent.style.alignItems = "center";
  modalContent.style.justifyContent = "center";
  modal.appendChild(modalContent);
  document.body.appendChild(modal);

  const loader = document.createElement("div");
  loader.innerHTML = "Đang kiểm tra video, xin vui lòng đợi...";
  loader.style.fontSize = "20px";
  loader.style.color = "#000";
  loader.style.textAlign = "center";
  loader.style.marginTop = "3%";

  // Tạo vòng xoay
  const spinner = document.createElement("div");
  spinner.style.border = "8px solid #f3f3f3"; // Màu nền
  spinner.style.borderTop = "8px solid #3498db"; // Màu vòng xoay
  spinner.style.borderRadius = "50%";
  spinner.style.width = "50px";
  spinner.style.height = "50px";
  spinner.style.animation = "spin 1s linear infinite"; // Hiệu ứng xoay

  // Áp dụng CSS cho hiệu ứng xoay
  const style = document.createElement("style");
  style.innerHTML = `
    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
`;
  document.head.appendChild(style);

  // Thêm vòng xoay vào loader
  modalContent.appendChild(spinner);
  modalContent.appendChild(loader);

  var url = "https://firefly-genuine-adequately.ngrok-free.app/process"; // URL API backend
  //call API
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
    if (!response.ok) {
      throw new Error("Kiểm tra thất bại"); // thay bằng thông báo lỗi của api
    }
    let data = await response.json();

    // Remove loader after data is loaded
    if (modalContent.contains(loader) && modalContent.contains(spinner)) {
      modalContent.removeChild(loader);
      modalContent.removeChild(spinner);
    }

    // Show success message
    const successMessage = document.createElement("div");
    successMessage.innerHTML = `Dữ liệu tải thành công🎆! <br> ${
      data.result ? "Đây là video " + data.result : "Không có thông tin"
    }`;
    // thay bằng message api trả về
    successMessage.style.fontSize = "30px";
    successMessage.style.color = "green";
    successMessage.style.textAlign = "center";
    successMessage.style.marginTop = "15px"; // Adjust margin for better alignment
    modalContent.appendChild(successMessage);

    // Create tag a contain link form server
    const linkMessage = document.createElement("a");
    linkMessage.href = data.output_path; // Thay bằng link của backend
    linkMessage.target = "_blank";
    linkMessage.innerHTML = "Kiểm tra video ngay tại đây❤️!";
    linkMessage.style.color = "blue";
    linkMessage.style.textDecoration = "underline";
    linkMessage.style.fontSize = "30px";
    linkMessage.style.display = "block";
    successMessage.style.marginTop = "10px";

    modalContent.appendChild(linkMessage);
  } catch (error) {
    console.error("Lỗi:", error);
    // Remove loader after data is loaded or failed
    if (modalContent.contains(loader) && modalContent.contains(spinner)) {
      modalContent.removeChild(loader);
      modalContent.removeChild(spinner);
    }

    // Create and display failure message
    const errorMessage = document.createElement("div");
    errorMessage.innerHTML = "Kiểm tra thất bại, vui lòng thử lại🥲🥲🥲!";
    errorMessage.style.fontSize = "30px";
    errorMessage.style.color = "red";
    errorMessage.style.textAlign = "center";
    errorMessage.style.marginTop = "20px";
    modalContent.appendChild(errorMessage);
  }

  // Create close button
  const closeButton = document.createElement("button");
  closeButton.innerHTML = "X"; // Đảm bảo dấu "X" được hiển thị
  closeButton.style.width = "40px"; // Đặt kích thước nút
  closeButton.style.height = "40px"; // Đặt chiều cao nút
  closeButton.style.fontSize = "20px"; // Đảm bảo chữ "X" đủ lớn
  closeButton.style.position = "absolute";
  closeButton.style.top = "10px";
  closeButton.style.right = "10px";
  closeButton.style.color = "#000"; // Màu chữ "X" (trắng)
  closeButton.style.backgroundColor = "transparent"; // Nền trong suốt
  closeButton.style.border = "2px solid #272727"; // Viền nút
  closeButton.style.cursor = "pointer";
  closeButton.style.borderRadius = "50%"; // Viền tròn
  closeButton.style.padding = "0"; // Xóa padding thừa
  closeButton.style.zIndex = "10000"; // Ensure the button is on top of other elements

  closeButton.addEventListener("click", function () {
    document.body.removeChild(modal);
    modal = null; // Clear modal reference
  });

  modalContent.appendChild(closeButton);
  //modalContent.appendChild(iframe);
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

window.addEventListener("load", function () {
  //this.window.location.reload();
  setInterval(runExtension, 1000);
});
