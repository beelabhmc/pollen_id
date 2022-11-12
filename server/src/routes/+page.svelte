<script lang="ts">
	import { ProgressIndicator, ProgressStep, Button } from 'carbon-components-svelte';

	import Upload from '../components/upload.svelte';
    import Select from '../components/select.svelte';

	let currentIndex = 1;

	type states = 'upload' | 'select' | 'classify' | 'results';

	let state: states = 'upload';

	function setState(newState: states) {
		state = newState;
		switch (state) {
			case 'upload':
				currentIndex = 0;
				break;
			case 'select':
				currentIndex = 1;
				break;
			case 'classify':
				currentIndex = 2;
				break;
			case 'results':
				currentIndex = 3;
				break;
		}
	}
	setState(state);

    let images: {name: string, url: string, pollen?: {species?: string, box: {x: number, y: number, w: number, h: number}}}[] = [];
</script>

<ProgressIndicator preventChangeOnClick bind:currentIndex>
	<ProgressStep
		complete={currentIndex > 0}
		label="Upload Images"
		description="The progress indicator will listen for clicks on the steps"
	/>
	<ProgressStep
		complete={currentIndex > 1}
		label="Select Pollen"
		description="The progress indicator will listen for clicks on the steps"
	/>
	<ProgressStep
		complete={currentIndex > 2}
		label="Identify Species"
		description="The progress indicator will listen for clicks on the steps"
	/>
	<ProgressStep
		complete={currentIndex > 3}
		label="Export Results"
		description="The progress indicator will listen for clicks on the steps"
	/>
</ProgressIndicator>

{#if state == 'upload'}
    <Upload bind:images />
    <Button on:click={() => setState('select')} disabled={images.length == 0}>Next</Button>
{:else if state == 'select'}
    <Select bind:images />
    <br>
    <Button on:click={() => setState('classify')}>Next</Button>
{:else if state == 'classify'}
    <p>Identify Species</p>
    <Button on:click={() => setState('results')}>Next</Button>
{:else if state == 'results'}
    <p>Export Results</p>
{/if}