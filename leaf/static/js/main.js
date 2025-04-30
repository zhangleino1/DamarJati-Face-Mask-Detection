$(document).ready(function() {
    // 获取元素
    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');
    const previewContainer = $('#preview-container');
    const previewImage = $('#preview-image');
    const uploadIcon = $('#upload-icon');
    const analyzeBtn = $('#analyze-btn');
    const resetBtn = $('#reset-btn');
    const uploadForm = $('#upload-form');
    const loadingIndicator = $('#loading');
    const resultContainer = $('#result-container');
    const predictionResult = $('#prediction-result');
    const predictionConfidence = $('#prediction-confidence');
    const probabilitiesContainer = $('#probabilities-container');
    const placeholder = $('#placeholder');
    const errorContainer = $('#error-container');
    const errorMessage = $('#error-message');
    
    // 点击上传区域触发文件选择
    dropzone.on('click', function() {
        fileInput.click();
    });
    
    // 拖放功能
    dropzone.on('dragover', function(e) {
        e.preventDefault();
        dropzone.addClass('dragover');
    });
    
    dropzone.on('dragleave', function() {
        dropzone.removeClass('dragover');
    });
    
    dropzone.on('drop', function(e) {
        e.preventDefault();
        dropzone.removeClass('dragover');
        
        if (e.originalEvent.dataTransfer.files.length) {
            handleFile(e.originalEvent.dataTransfer.files[0]);
        }
    });
    
    // 文件选择事件
    fileInput.on('change', function() {
        if (fileInput[0].files.length) {
            handleFile(fileInput[0].files[0]);
        }
    });
    
    // 处理选择的文件
    function handleFile(file) {
        // 检查文件类型
        if (!file.type.match('image.*')) {
            showError('请选择图片文件 (JPG, PNG 等)');
            return;
        }
        
        // 预览图片
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.attr('src', e.target.result);
            uploadIcon.addClass('d-none');
            previewContainer.removeClass('d-none');
            analyzeBtn.prop('disabled', false);
        };
        reader.readAsDataURL(file);
        
        // 隐藏错误信息
        errorContainer.addClass('d-none');
    }
    
    // 重置按钮
    resetBtn.on('click', function() {
        resetUI();
    });
    
    // 重置UI
    function resetUI() {
        uploadForm[0].reset();
        uploadIcon.removeClass('d-none');
        previewContainer.addClass('d-none');
        analyzeBtn.prop('disabled', true);
        loadingIndicator.addClass('d-none');
        resultContainer.addClass('d-none');
        placeholder.removeClass('d-none');
        errorContainer.addClass('d-none');
    }
    
    // 显示错误信息
    function showError(message) {
        errorMessage.text(message);
        errorContainer.removeClass('d-none');
        loadingIndicator.addClass('d-none');
        placeholder.addClass('d-none');
    }
    
    // 提交表单
    uploadForm.on('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput[0].files.length) {
            showError('请先选择一张图片');
            return;
        }
        
        // 显示加载指示器
        loadingIndicator.removeClass('d-none');
        resultContainer.addClass('d-none');
        placeholder.addClass('d-none');
        errorContainer.addClass('d-none');
        
        // 准备表单数据
        const formData = new FormData();
        formData.append('file', fileInput[0].files[0]);
        
        // 发送请求
        $.ajax({
            url: '/predict/',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // 隐藏加载指示器
                loadingIndicator.addClass('d-none');
                
                // 显示结果
                predictionResult.text(response.prediction);
                predictionConfidence.text((response.probability * 100).toFixed(2) + '%');
                
                // 显示所有概率
                probabilitiesContainer.empty();
                
                // 将概率排序并显示为进度条
                const probabilities = response.all_probabilities;
                Object.entries(probabilities).forEach(([name, prob]) => {
                    const percentage = (prob * 100).toFixed(2);
                    const colorClass = percentage > 50 ? 'bg-success' : 
                                      percentage > 20 ? 'bg-info' : 
                                      percentage > 10 ? 'bg-warning' : 'bg-danger';
                    
                    const progressBar = `
                        <div class="mb-3">
                            <div class="progress">
                                <div class="progress-bar ${colorClass}" role="progressbar" 
                                     style="width: ${percentage}%" 
                                     aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                                    ${name}: ${percentage}%
                                </div>
                            </div>
                        </div>
                    `;
                    
                    probabilitiesContainer.append(progressBar);
                });
                
                resultContainer.removeClass('d-none');
            },
            error: function(xhr) {
                loadingIndicator.addClass('d-none');
                
                let errorMsg = '请求处理过程中出现错误';
                if (xhr.responseJSON && xhr.responseJSON.detail) {
                    errorMsg = xhr.responseJSON.detail;
                }
                
                showError(errorMsg);
            }
        });
    });
});
