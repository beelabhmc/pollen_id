<script lang="ts">
	import {
		FileUploaderDropContainer,
		Grid,
		Row,
		Column,
		Tile,
		ImageLoader,
		DataTable,
		Toolbar,
		ToolbarContent,
		ToolbarBatchActions,
		ToolbarSearch,
		NumberInput,
		Button
	} from 'carbon-components-svelte';

	import { white } from '@carbon/colors';

	import Edit from 'carbon-icons-svelte/lib/Edit.svelte';
	import Save from 'carbon-icons-svelte/lib/Save.svelte';

	let scalingFactorInput = 10;

	function readURL(f: File) {
		var reader = new FileReader();

		reader.onload = function (e) {
			if (e.target) {
				const img = new Image();
				img.src = e.target.result as string;
				img.onload = () => {
					images = [
						...images,
						{
							name: f.name,
							img: img,
							pollen: [],
							scaling_factor: scalingFactorInput
						}
					];
				};
			}
		};

		reader.readAsDataURL(f);
	}

	export let images: {
		name: string;
		img: HTMLImageElement;
		scaling_factor: number;
		pollen: {}[];
	}[] = [];

	let editing = false;
	let selectedRowIds: any[] = [];
</script>

<Grid>
	<Row>
		{#if images.length > 0}
			<Column padding>
				<DataTable
					expandable
					batchSelection
					bind:selectedRowIds
					title="Images"
					headers={[
						{ key: 'name', value: 'Name' },
						{ key: 'resolution', value: 'Resolution' },
						{ key: 'scaling_factor', value: 'Pixels/Micron' },
						{ key: 'size', value: 'Size' }
					]}
					rows={images.map((image) => {
						return {
							id: image.name,
							name: image.name,
							resolution: `${image.img.width} x ${image.img.height}`,
							size: ((image.img.src.length * (3 / 4)) / 1000000).toFixed(2) + ' MB',
							scaling_factor: image.scaling_factor,
							img: image.img
						};
					})}
				>
					<Toolbar>
						<ToolbarContent>
							{#if editing}
								<ToolbarBatchActions
									on:cancel={() => {
										editing = false;
									}}
								>
									<NumberInput
										hideLabel
										label="scaling_factor"
										hideSteppers
										bind:value={scalingFactorInput}
									/>
									<p style={`color: ${white}`}>&nbsp; pixels/micron</p>
									<Button
										icon={Save}
										disabled={selectedRowIds.length === 0}
										on:click={() => {
											editing = false;
											for (let i = 0; i < images.length; i++) {
												if (selectedRowIds.includes(images[i].name)) {
													images[i].scaling_factor = scalingFactorInput;
												}
											}
										}}>Save</Button
									>
								</ToolbarBatchActions>
							{:else}
								<Button
									icon={Edit}
									disabled={selectedRowIds.length === 0}
									on:click={() => {
										editing = true;
									}}>Edit Scaling Factor</Button
								>
							{/if}
						</ToolbarContent>
					</Toolbar>
					<svelte:fragment slot="expanded-row" let:row>
						<img src={row.img.src} height="100" alt={row.name} />
					</svelte:fragment>
				</DataTable>
			</Column>
		{/if}
	</Row>
	<Row>
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
	</Row>
</Grid>
