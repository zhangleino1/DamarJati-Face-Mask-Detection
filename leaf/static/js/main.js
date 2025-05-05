$(document).ready(function() {
    // 文件上传区域相关功能
    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');
    const previewContainer = $('#preview-container');
    const previewImage = $('#preview-image');
    const uploadIcon = $('#upload-icon');
    const analyzeBtn = $('#analyze-btn');
    const resetBtn = $('#reset-btn');
    const resultContainer = $('#result-container');
    const placeholder = $('#placeholder');
    const loading = $('#loading');
    const errorContainer = $('#error-container');
    const errorMessage = $('#error-message');
    
    // 标记变量，用于防止递归触发
    let isFileInputTriggered = false;
    
    // 点击上传区域触发文件选择
    dropzone.on('click', function(e) {
        // 防止事件冒泡
        e.stopPropagation();
        
        // 只有当点击的不是预览图片且不是处于文件选择触发状态时才触发文件选择
        if (!$(e.target).closest('#preview-container').length && !isFileInputTriggered) {
            isFileInputTriggered = true;
            fileInput.trigger('click');
            // 设置短暂延时后重置标记变量
            setTimeout(function() {
                isFileInputTriggered = false;
            }, 100);
        }
    });
    
    // 处理文件拖放
    dropzone.on('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.addClass('dragover');
    });
    
    dropzone.on('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.removeClass('dragover');
    });
    
    dropzone.on('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.removeClass('dragover');
        
        if (e.originalEvent.dataTransfer.files.length) {
            fileInput[0].files = e.originalEvent.dataTransfer.files;
            handleFileSelect(e.originalEvent.dataTransfer.files[0]);
        }
    });
    
    // 处理文件选择
    fileInput.on('change', function(e) {
        if (fileInput[0].files.length) {
            handleFileSelect(fileInput[0].files[0]);
        }
    });
    
    // 处理选择的文件
    function handleFileSelect(file) {
        // 检查文件类型
        if (!file.type.match('image.*')) {
            showError('请选择图片文件（JPG、PNG）');
            return;
        }
        
        // 显示预览
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.attr('src', e.target.result);
            uploadIcon.addClass('d-none');
            previewContainer.removeClass('d-none');
            analyzeBtn.prop('disabled', false);
        };
        reader.readAsDataURL(file);
    }
    
    // 重置功能
    resetBtn.on('click', function() {
        fileInput.val('');
        previewImage.attr('src', '#');
        uploadIcon.removeClass('d-none');
        previewContainer.addClass('d-none');
        analyzeBtn.prop('disabled', true);
        resultContainer.addClass('d-none');
        placeholder.removeClass('d-none');
        errorContainer.addClass('d-none');
    });
    
    // 提交表单进行分析
    $('#upload-form').on('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput[0].files.length) {
            showError('请先选择一张图片');
            return;
        }
        
        // 显示加载中状态
        placeholder.addClass('d-none');
        resultContainer.addClass('d-none');
        errorContainer.addClass('d-none');
        loading.removeClass('d-none');
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('file', fileInput[0].files[0]);  // Changed from 'image' to 'file' to match backend
        
        // 发送AJAX请求
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                loading.addClass('d-none');
                
                // Update response handling to match the actual backend response format
                displayResults(response);
            },
            error: function(xhr) {
                loading.addClass('d-none');
                showError('服务器请求失败，请稍后重试: ' + (xhr.responseJSON?.detail || '未知错误'));
            }
        });
    });
    
    // 显示结果
    function displayResults(data) {
        $('#prediction-result').text(data.prediction);
        const probability = (data.probability * 100).toFixed(2);
        $('#prediction-confidence').text(probability + '%');
        
        // 显示所有类别的概率
        const probContainer = $('#probabilities-container');
        probContainer.empty();
        
        // Use all_probabilities from the backend response
        Object.entries(data.all_probabilities).forEach(function([key, value]) {
            const probabilityPercent = (value * 100).toFixed(2);
            const progressBar = $('<div class="progress mb-2" style="height: 25px;">' +
                '<div class="progress-bar" role="progressbar" style="width: ' + probabilityPercent + '%;" ' +
                'aria-valuenow="' + probabilityPercent + '" aria-valuemin="0" aria-valuemax="100">' +
                key + ': ' + probabilityPercent + '%</div></div>');
            probContainer.append(progressBar);
        });
        
        resultContainer.removeClass('d-none');
    }
    
    // 显示错误信息
    function showError(message) {
        errorMessage.text(message);
        errorContainer.removeClass('d-none');
        loading.addClass('d-none');
    }
});
