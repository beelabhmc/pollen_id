<script lang="ts">
	import {
		FileUploaderDropContainer,
		Grid,
		Row,
		Column,
		Tile,
		ImageLoader
	} from 'carbon-components-svelte';

	function readURL(f: File) {
		var reader = new FileReader();

		reader.onload = function (e) {
			if (e.target) {
				images = [
					...images,
					{
						name: f.name,
						url: e.target.result as string
					}
				];
			}
		};

		reader.readAsDataURL(f);
	}

	export let images: { name: string; url: string }[] = [];
</script>

<Grid>
	<Row>
		{#if images.length > 0}
			{#each images as file, i}
				<Column padding>
					<Tile>
						<ImageLoader src={file.url} />
						{file.name}
					</Tile>
				</Column>
			{/each}
		{:else}
			<Column padding>
				<FileUploaderDropContainer
					multiple
					labelText="Drag and drop files here or click to upload (only JPG and PNG files are accepted)"
					accept={['.jpg', '.jpeg', '.png']}
					on:change={(e) => {
						e.detail.forEach((f) => {
							readURL(f);
						});
					}}
				/>
			</Column>
		{/if}
	</Row>
</Grid>
