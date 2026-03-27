/**
 * Electric Load Forecasting - UI Enhancement Scripts
 */

// Wait for document to be ready
$(document).ready(function() {
    // Initialize tooltips
    initTooltips();
    
    // Add animation to plot containers
    animatePlotContainers();
    
    // Enhance the K-value slider with immediate feedback
    enhanceSlider();
    
    // Add greeting message
    showWelcomeMessage();
    
    // Add smooth tab transitions
    smoothTabTransitions();
    
    // Enhance the loading indicator
    enhanceLoadingIndicator();
    
    // Add CSS class for pulse animation
    $('<style>')
        .prop('type', 'text/css')
        .html(`
            .pulse {
                animation: pulse 0.3s ease-in-out;
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }
        `)
        .appendTo('head');
});

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            container: 'body'
        });
    });
}

/**
 * Add subtle animation to plot containers
 */
function animatePlotContainers() {
    $('.plot-container').each(function(index) {
        $(this).css('opacity', 0);
        $(this).delay(100 * index).animate({
            opacity: 1
        }, 500);
    });
}

/**
 * Enhance the K-value slider with immediate visual feedback
 */
function enhanceSlider() {
    $('#kSlider').on('input', function() {
        const kValue = $(this).val();
        $('#kValue').text(kValue).removeClass().addClass('badge bg-primary');
        
        // Add visual pulse effect
        $('#kValue').addClass('pulse');
        setTimeout(function() {
            $('#kValue').removeClass('pulse');
        }, 300);
    });
}

/**
 * Show welcome message with usage tips
 */
function showWelcomeMessage() {
    const now = new Date();
    const hour = now.getHours();
    let greeting = "Good morning";
    
    if (hour >= 12 && hour < 18) {
        greeting = "Good afternoon";
    } else if (hour >= 18) {
        greeting = "Good evening";
    }
    
    // Check if this is the user's first visit
    if (!localStorage.getItem('visited')) {
        showToast(
            `${greeting}! Welcome to the Electric Load Forecasting Dashboard.`,
            `Select a city and explore electricity consumption patterns. Use the controls on the left to adjust parameters.`,
            'info',
            8000
        );
        localStorage.setItem('visited', 'true');
    } else {
        showToast(
            `${greeting}!`,
            `Welcome back to the Electric Load Forecasting Dashboard.`,
            'info',
            3000
        );
    }
}

/**
 * Create and show a toast notification
 */
function showToast(title, message, type = 'info', duration = 5000) {
    // Create toast container if it doesn't exist
    if ($('#toastContainer').length === 0) {
        $('body').append('<div id="toastContainer" class="toast-container position-fixed top-0 end-0 p-3"></div>');
    }
    
    // Create unique ID for this toast
    const toastId = 'toast-' + Date.now();
    
    // Create toast HTML
    const toast = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-${type} text-white">
                <i class="fas fa-info-circle me-2"></i>
                <strong class="me-auto">${title}</strong>
                <small>just now</small>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    // Add toast to container
    $('#toastContainer').append(toast);
    
    // Initialize and show the toast
    const toastElement = document.getElementById(toastId);
    const bsToast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: duration
    });
    bsToast.show();
    
    // Remove toast from DOM after it's hidden
    $(toastElement).on('hidden.bs.toast', function() {
        $(this).remove();
    });
}

/**
 * Add smooth transitions between tabs
 */
function smoothTabTransitions() {
    $('.nav-link').on('click', function() {
        const targetId = $(this).attr('href');
        
        // Fade out all tabs
        $('.tab-pane').css('opacity', 0);
        
        // Fade in the target tab
        setTimeout(function() {
            $(targetId).css('opacity', 1);
        }, 150);
    });
    
    // Set initial opacity for all tabs
    $('.tab-pane').css({
        'opacity': 0,
        'transition': 'opacity 0.15s ease-in-out'
    });
    
    // Set initial active tab to visible
    $('.tab-pane.active').css('opacity', 1);
}

/**
 * Enhance the loading indicator with additional feedback
 */
function enhanceLoadingIndicator() {
    // Store original button text
    $('.btn').each(function() {
        $(this).data('original-text', $(this).html());
    });
    
    // Show loading indicator when buttons are clicked
    $('.btn').on('click', function() {
        const btn = $(this);
        const originalText = btn.data('original-text');
        
        // Don't process if already loading
        if (btn.hasClass('loading')) return;
        
        // Add loading state
        btn.addClass('loading')
           .attr('disabled', true)
           .html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...');
        
        // Simulate API call and reset button (replace with actual API calls)
        setTimeout(function() {
            btn.removeClass('loading')
               .attr('disabled', false)
               .html(originalText);
        }, 1000);
    });
} 