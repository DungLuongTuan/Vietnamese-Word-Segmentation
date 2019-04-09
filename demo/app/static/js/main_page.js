$(document).ready(function() {
	let $input = $('#input');
	let $processing = $('.processing')
	var $send_button = $('#send_button');

	function predictSentiment() {
		let text = $input.val();
		$processing.css('display', 'block');
		$.ajax({
			url: '/',
			data: {'req': text},
			type: 'POST',
			success: function(response) {
				$processing.css('display', 'none')
				var res = JSON.parse(response)
				var $output = $('#output');
				$output.val(res.segmented_text);
			},
			error: function(error) {
				console.log(error);
			}
		});
	}

	$send_button.click(function(){
		predictSentiment();
	})
})