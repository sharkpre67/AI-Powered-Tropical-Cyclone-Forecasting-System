const currentMonth = new Date().getMonth() + 1;
// Gán giá trị cho select
document.getElementById('month').value = currentMonth;

const nowUTC = new Date();
const currentHourUTC = nowUTC.getUTCHours();

// Gán giá trị vào input
document.getElementById('hour').value = currentHourUTC;

// Khởi tạo bản đồ
const map = L.map('map').setView([16.0, 108.0], 6);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Biến toàn cục
let selectedLatLng = null;
let marker = null;
let probabilityResult = null;

// Kiểm tra trạng thái server khi tải trang
checkServerStatus();

// Thêm marker khi click vào bản đồ
map.on('click', function (e) {
    selectedLatLng = e.latlng;

    // Cập nhật giá trị trong form
    document.getElementById('latitude').value = selectedLatLng.lat.toFixed(4);
    document.getElementById('longitude').value = selectedLatLng.lng.toFixed(4);

    // Xóa marker cũ (nếu có)
    if (marker) {
        map.removeLayer(marker);
    }

    // Thêm marker mới
    marker = L.marker(selectedLatLng).addTo(map)
        .bindPopup(`Vị trí đã chọn:<br>Vĩ độ: ${selectedLatLng.lat.toFixed(4)}<br>Kinh độ: ${selectedLatLng.lng.toFixed(4)}`)
        .openPopup();

    // Reset kết quả trước đó
    resetResults();
});

// Cập nhật vị trí marker khi nhập tọa độ thủ công
document.getElementById('latitude').addEventListener('change', updateMarkerFromInput);
document.getElementById('longitude').addEventListener('change', updateMarkerFromInput);

function updateMarkerFromInput() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lng = parseFloat(document.getElementById('longitude').value);

    if (!isNaN(lat) && !isNaN(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
        selectedLatLng = L.latLng(lat, lng);

        // Xóa marker cũ (nếu có)
        if (marker) {
            map.removeLayer(marker);
        }

        // Thêm marker mới
        marker = L.marker(selectedLatLng).addTo(map)
            .bindPopup(`Vị trí đã chọn:<br>Vĩ độ: ${selectedLatLng.lat.toFixed(4)}<br>Kinh độ: ${selectedLatLng.lng.toFixed(4)}`)
            .openPopup();

        // Di chuyển bản đồ đến vị trí mới
        map.setView(selectedLatLng, 8);

        // Reset kết quả trước đó
        resetResults();
    }
}

// Xử lý tự động phân loại bão khi nhập tốc độ gió
document.getElementById('windSpeed').addEventListener('input', function () {
    const windSpeed = parseFloat(this.value);

    if (!isNaN(windSpeed) && windSpeed >= 0) {
        // Tự động chọn phân loại bão dựa trên tốc độ gió
        let natureValue = 'TD';
        let natureText = '';

        if (windSpeed < 34) {
            natureValue = 'TD';
            natureText = 'Áp thấp nhiệt đới';
        } else if (windSpeed < 48) {
            natureValue = 'TS';
            natureText = 'Bão nhiệt đới';
        } else if (windSpeed < 64) {
            natureValue = 'STS';
            natureText = 'Bão nhiệt đới dữ dội';
        } else if (windSpeed < 100) {
            natureValue = 'TY';
            natureText = 'Bão';
        } else {
            natureValue = 'STY';
            natureText = 'Siêu bão';
        }

        // Cập nhật giá trị select
        document.getElementById('nature').value = natureValue;

        // Hiển thị thông báo
        const explanationDiv = document.getElementById('natureExplanation');
        explanationDiv.innerHTML = `
                    <strong>Phân loại tự động:</strong> ${natureText} - 
                    Tốc độ gió ${windSpeed} knots thuộc phân loại ${natureValue}
                `;
    }
});

// Xử lý dự đoán khả năng có bão
document.getElementById('predictProbability').addEventListener('click', predictProbability);

async function predictProbability() {
    if (!selectedLatLng) {
        alert('Vui lòng chọn vị trí trên bản đồ hoặc nhập tọa độ trước.');
        return;
    }

    const lat = selectedLatLng.lat;
    const lng = selectedLatLng.lng;

    // Hiển thị loading và ẩn kết quả cũ
    const loadingDiv = document.getElementById('probabilityLoading');
    const resultDiv = document.getElementById('probabilityResult');
    const progressBar = document.getElementById('probabilityProgress');
    const progressText = document.getElementById('probabilityProgressText');

    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    // Hiệu ứng progress bar
    simulateProgress(progressBar, progressText, [
        "Đang kết nối đến máy chủ...",
        "Đang tải mô hình dự đoán...",
        "Đang phân tích dữ liệu khí tượng...",
        "Đang tính toán xác suất..."
    ]);

    try {
        const response = await fetch('/predict_prob', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                lat: lat,
                lon: lng
            })
        });

        const data = await response.json();

        // Ẩn loading
        loadingDiv.classList.add('hidden');

        if (data.success) {
            probabilityResult = data;

            // Hiển thị kết quả
            let probabilityClass, alertText;
            if (data.probability > 70) {
                probabilityClass = 'high-probability';
                alertText = 'CẢNH BÁO: Khả năng có bão cao!';
            } else if (data.probability > 30) {
                probabilityClass = 'medium-probability';
                alertText = 'CẢNH BÁO: Khả năng có bão trung bình!';
            } else {
                probabilityClass = 'low-probability';
                alertText = 'Khả năng có bão thấp';
            }

            resultDiv.innerHTML = `
                        <div class="probability-display ${probabilityClass}">
                            <h3>${alertText}</h3>
                            <p>Xác suất có bão: <strong>${data.probability}%</strong></p>
                            <p>Tọa độ: ${data.lat.toFixed(4)}, ${data.lon.toFixed(4)}</p>
                        </div>
                    `;

            // Kích hoạt hoặc vô hiệu hóa nút dự đoán đường đi
            const predictTrackBtn = document.getElementById('predictTrack');

            if (data.alert) {
                predictTrackBtn.disabled = false;
                document.getElementById('stormParamsForm').querySelector('.alert-info').textContent =
                    'Khả năng có bão cao. Vui lòng nhập thông số bão để dự đoán đường đi.';
            } else {
                predictTrackBtn.disabled = true;
                document.getElementById('stormParamsForm').querySelector('.alert-info').textContent =
                    'Khả năng có bão thấp. Không cần dự đoán đường đi.';
            }

            resultDiv.classList.remove('hidden');

        } else {
            resultDiv.innerHTML = `<div class="alert-message alert-danger">Lỗi: ${data.message}</div>`;
            resultDiv.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Error:', error);
        loadingDiv.classList.add('hidden');
        resultDiv.innerHTML = `<div class="alert-message alert-danger">Lỗi kết nối: ${error.message}</div>`;
        resultDiv.classList.remove('hidden');
    }
}

// Xử lý dự đoán đường đi và cường độ
document.getElementById('predictTrack').addEventListener('click', predictTrack);

async function predictTrack() {
    if (!probabilityResult || !probabilityResult.alert) {
        alert('Không thể dự đoán đường đi vì khả năng có bão thấp.');
        return;
    }

    // Lấy giá trị từ form
    const windSpeed = document.getElementById('windSpeed').value;
    const pressure = document.getElementById('pressure').value;
    const nature = document.getElementById('nature').value;
    const month = document.getElementById('month').value;
    const hour = document.getElementById('hour').value;

    // Kiểm tra dữ liệu đầu vào
    if (!windSpeed || !pressure) {
        alert('Vui lòng nhập đầy đủ tốc độ gió và áp suất.');
        return;
    }

    // Hiển thị loading
    const loadingDiv = document.getElementById('trackLoading');
    const resultDiv = document.getElementById('trackResult');
    const progressBar = document.getElementById('trackProgress');
    const progressText = document.getElementById('trackProgressText');

    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    // Hiệu ứng progress bar
    simulateProgress(progressBar, progressText, [
        "Đang khởi tạo mô hình dự báo...",
        "Đang xử lý dữ liệu đầu vào...",
        "Đang dự đoán đường đi...",
        "Đang tính toán cường độ...",
        "Đang tạo báo cáo kết quả..."
    ]);

    try {
        const response = await fetch('/predict_track', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                lat: selectedLatLng.lat,
                lon: selectedLatLng.lng,
                wind: parseFloat(windSpeed),
                pressure: parseFloat(pressure),
                nature: nature,
                month: parseInt(month),
                hour: parseInt(hour)
            })
        });

        const data = await response.json();

        // Ẩn loading
        loadingDiv.classList.add('hidden');

        if (data.error) {
            resultDiv.innerHTML = `<div class="alert-message alert-danger">Lỗi: ${data.error}</div>`;
            resultDiv.classList.remove('hidden');
            return;
        }

        // Hiển thị kết quả
        displayTrackResult(data);

    } catch (error) {
        console.error('Error:', error);
        loadingDiv.classList.add('hidden');
        resultDiv.innerHTML = `<div class="alert-message alert-danger">Lỗi kết nối: ${error.message}</div>`;
        resultDiv.classList.remove('hidden');
    }
}

function displayTrackResult(data) {
    const resultDiv = document.getElementById('trackResult');

    // Tạo bảng kết quả
    let tableHTML = `
                <h3>📈 Dự báo đường đi và cường độ bão</h3>
                <div class="alert-message alert-success">
                    <p><strong>Thông tin bão hiện tại:</strong> Vị trí (${data.current.lat.toFixed(2)}, ${data.current.lon.toFixed(2)}), 
                    Gió: ${data.current.wind} knots, Áp suất: ${data.current.pressure} hPa, Phân loại: ${data.current.nature}</p>
                </div>
                <table class="forecast-table">
                    <thead>
                        <tr>
                            <th>Thời gian (giờ)</th>
                            <th>Vị trí (Lat, Lon)</th>
                            <th>Tốc độ gió (knots)</th>
                            <th>Áp suất (hPa)</th>
                            <th>Tốc độ di chuyển (km/h)</th>
                            <th>Hướng di chuyển</th>
                            <th>Phân loại</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

    data.forecasts.forEach(forecast => {
        tableHTML += `
                    <tr>
                        <td>+${forecast.hours}</td>
                        <td>${forecast.lat}, ${forecast.lon}</td>
                        <td>${forecast.wind}</td>
                        <td>${forecast.pressure}</td>
                        <td>${forecast.speed}</td>
                        <td>${forecast.direction}</td>
                        <td>${forecast.nature}</td>
                    </tr>
                `;
    });

    tableHTML += `
                    </tbody>
                </table>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #ffeb3b;"></div>
                        <span>TD - Áp thấp nhiệt đới</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #ff9800;"></div>
                        <span>TS - Bão nhiệt đới</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #f44336;"></div>
                        <span>STS - Bão nhiệt đới dữ dội</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #d32f2f;"></div>
                        <span>TY - Bão</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #b71c1c;"></div>
                        <span>STY - Siêu bão</span>
                    </div>
                </div>
            `;

    resultDiv.innerHTML = tableHTML;
    resultDiv.classList.remove('hidden');

    // Vẽ đường đi trên bản đồ
    drawTrackOnMap(data);
}

function drawTrackOnMap(data) {
    // Xóa các layer cũ (nếu có)
    if (window.trackLayers) {
        window.trackLayers.forEach(layer => map.removeLayer(layer));
    }

    window.trackLayers = [];

    // Tọa độ điểm bắt đầu
    const startPoint = [data.current.lat, data.current.lon];

    // Tạo mảng tọa độ đường đi
    const trackCoords = [startPoint];
    data.forecasts.forEach(forecast => {
        trackCoords.push([forecast.lat, forecast.lon]);
    });

    // Vẽ đường đi
    const trackLine = L.polyline(trackCoords, {
        color: 'red',
        weight: 4,
        opacity: 0.7,
        dashArray: '10, 10'
    }).addTo(map);

    window.trackLayers.push(trackLine);

    // Thêm marker cho điểm bắt đầu
    const startMarker = L.marker(startPoint)
        .addTo(map)
        .bindPopup(`<b>Vị trí hiện tại</b><br>Gió: ${data.current.wind} knots<br>Áp suất: ${data.current.pressure} hPa<br>Phân loại: ${data.current.nature}`);

    window.trackLayers.push(startMarker);

    // Thêm marker cho các điểm dự báo
    data.forecasts.forEach(forecast => {
        const point = [forecast.lat, forecast.lon];
        let iconColor;

        // Chọn màu marker dựa trên phân loại bão
        switch (forecast.nature) {
            case 'TD': iconColor = '#ffeb3b'; break;
            case 'TS': iconColor = '#ff9800'; break;
            case 'STS': iconColor = '#f44336'; break;
            case 'TY': iconColor = '#d32f2f'; break;
            case 'STY': iconColor = '#b71c1c'; break;
            default: iconColor = '#757575';
        }

        const forecastIcon = L.divIcon({
            className: 'forecast-marker',
            html: `<div style="background-color: ${iconColor}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white;"></div>`,
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        });

        const forecastMarker = L.marker(point, { icon: forecastIcon })
            .addTo(map)
            .bindPopup(`
                        <b>Dự báo sau ${forecast.hours} giờ</b><br>
                        Vị trí: ${forecast.lat}, ${forecast.lon}<br>
                        Gió: ${forecast.wind} knots<br>
                        Áp suất: ${forecast.pressure} hPa<br>
                        Tốc độ: ${forecast.speed} km/h<br>
                        Hướng: ${forecast.direction}<br>
                        Phân loại: ${forecast.nature}
                    `);

        window.trackLayers.push(forecastMarker);
    });

    // Fit map để hiển thị toàn bộ đường đi
    const trackGroup = L.featureGroup(window.trackLayers);
    map.fitBounds(trackGroup.getBounds().pad(0.1));
}

// Xử lý nút đặt lại form
document.getElementById('resetForm').addEventListener('click', function () {
    document.getElementById('windSpeed').value = '';
    document.getElementById('pressure').value = '';
    document.getElementById('nature').value = 'TD';
    document.getElementById('natureExplanation').innerHTML = 'Phân loại sẽ tự động thay đổi khi bạn nhập tốc độ gió';
});

// Reset form và kết quả
function resetResults() {
    // Reset kết quả xác suất
    document.getElementById('probabilityLoading').classList.add('hidden');
    document.getElementById('probabilityResult').classList.add('hidden');
    document.getElementById('probabilityResult').innerHTML = '';

    // Reset kết quả đường đi
    document.getElementById('trackLoading').classList.add('hidden');
    document.getElementById('trackResult').classList.add('hidden');
    document.getElementById('trackResult').innerHTML = '';

    // Vô hiệu hóa nút dự đoán đường đi
    document.getElementById('predictTrack').disabled = true;

    // Reset thông báo
    document.getElementById('stormParamsForm').querySelector('.alert-info').textContent =
        'Vui lòng chọn vị trí và dự đoán khả năng có bão trước.';

    // Xóa các layer trên bản đồ (nếu có)
    if (window.trackLayers) {
        window.trackLayers.forEach(layer => map.removeLayer(layer));
        window.trackLayers = [];
    }

    probabilityResult = null;
}

// Kiểm tra trạng thái server
async function checkServerStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();

        document.getElementById('statusServerText').textContent = 'Kết nối máy chủ: Hoạt động';
        document.getElementById('statusModelsText').textContent =
            `Mô hình: ${data.movement_models_loaded.length} mô hình đã tải`;

    } catch (error) {
        document.getElementById('statusServer').classList.remove('status-active');
        document.getElementById('statusServer').classList.add('status-inactive');
        document.getElementById('statusServerText').textContent = 'Kết nối máy chủ: Lỗi';
        document.getElementById('statusModelsText').textContent = 'Mô hình: Không kết nối được';
    }
}

// Hiệu ứng progress bar
function simulateProgress(progressBar, progressText, messages) {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
        }
        progressBar.style.width = `${progress}%`;

        // Thay đổi tin nhắn dựa trên tiến độ
        if (progress < 25 && messages[0]) {
            progressText.textContent = messages[0];
        } else if (progress < 50 && messages[1]) {
            progressText.textContent = messages[1];
        } else if (progress < 75 && messages[2]) {
            progressText.textContent = messages[2];
        } else if (messages[3]) {
            progressText.textContent = messages[3];
        }
    }, 300);
}

// Đặt giá trị mặc định cho tháng và giờ (hiện tại)
const now = new Date();
document.getElementById('month').value = now.getMonth() + 1;
document.getElementById('hour').value = Math.floor(now.getHours() / 6) * 6;