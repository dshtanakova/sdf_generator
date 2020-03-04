import torch
import torch.nn as nn
import torch.nn.functional as F

from math import tan, pi
from torchdiffeq import odeint_adjoint, odeint

from warnings import warn

__all__ = ['SDFRenderer']


class OdeintFunc(nn.Module):
    """
    A distance model wrapper, required for torchdiffeq
    """
    def __init__(self, dist_model, rendering_step, rays):
        super().__init__()
        self.dist_model = dist_model
        self.rendering_step = rendering_step
        self.rays = rays

    def forward(self, t, x):

        dists = self.dist_model(x.view(-1, 3))
        step = dists.view(-1, 1) * self.rendering_step

        return self.rays * step


class SDFRenderer(nn.Module):
    def __init__(
        self, image_size=256, background_color=(0., 0., 0.), near=0.1, far=255.,
        fov=90, rendering_step=0.95, hit_threshold=0.05, iterations=20, shading='distance'
    ):
        """
        :param image_size: toot vse yasno
        :param background_color: black is the new black
        :param near: maximum ray travel distance that makes any sense
        :param far: minimum ray travel distance that makes any sense
        :param rendering_step: fraction of distance returned by distance model, which ray covers in one iteration
        :param hit_threshold: hit threshold, if None, see what happens for yourself
        :param iterations: number of iterations made to render a batch of images
        """
        super().__init__()

        self.rendering_step = rendering_step
        self.hit_thresh = hit_threshold
        self.iterations = iterations
        self.shading = shading
        self.near = near
        self.far = far
        self.bsize = 1

        self.register_buffer('eye', torch.zeros(3, dtype=torch.float32))
        self.background_color = background_color

        assert isinstance(fov, (int, float)) and fov > 0
        assert isinstance(image_size, int) and image_size > 0

        self._fov = fov
        self._image_size = image_size
        self.__make_rays(fov, image_size)

    def __make_rays(self, fov, resolution):
        side = tan(pi * fov / 360)
        linspace = torch.linspace(-side, side, steps=resolution, device=self.eye.device)
        coordgrid = torch.stack(
            (*torch.meshgrid([linspace] * 2), torch.ones(resolution, resolution, device=self.eye.device)),
        -1).permute(1, 0, 2).flip(0)

        if 'rays' in self._buffers:
            self._buffers['rays'] = F.normalize(coordgrid.view(-1, 3), eps=1e-5)
        else:
            self.register_buffer('rays', F.normalize(coordgrid.view(-1, 3), eps=1e-5))

    @property
    def rendering_step(self):
        return self._rendering_step

    @rendering_step.setter
    def rendering_step(self, value):
        assert isinstance(value, float)
        assert 0 < value <= 1, "Don't"
        self._rendering_step = value

    @property
    def shading(self):
        return self._shading

    @shading.setter
    def shading(self, value):
        assert value in {'distance', 'distance_inverted', None}
        self._shading = value

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        assert isinstance(value, int) and value > 0

        if self._image_size != value:
            self._image_size = value
            self.__make_rays(self._fov, value)

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        assert isinstance(value, (float, int)) and value > 0

        if self._fov != value:
            self._fov = value
            self.__make_rays(value, self._image_size)

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        assert hasattr(value, '__len__') and len(value) == 3
        if '_background_color' in self._buffers:
            self._buffers['_background_color'] = torch.tensor(value, dtype=torch.float32, device=self.eye.device)
        else:
            self.register_buffer('_background_color', torch.tensor(value, dtype=torch.float32, device=self.eye.device))

    @staticmethod
    def look_at(eye, at=(0, 0, 0), up=(0, 0, 1)):
        """
        Computes rotation matrix, needed to rotate the coordinate system so that `eye` looks at `at`
        :param eye: camera position [B, 3]
        :param at: point to look at, tuple or [B, 3]
        :param up: up direction, tuple or [B, 3]
        :return: rotation matrices [B, 3, 3]
        """
        at = torch.tensor(at, dtype=torch.float32, device=eye.device)
        up = torch.tensor(up, dtype=torch.float32, device=eye.device)

        batch_size = eye.shape[0]
        if at.ndimension() == 1:
            at = at[None, :].repeat(batch_size, 1)
        if up.ndimension() == 1:
            up = up[None, :].repeat(batch_size, 1)

        view_vector = F.normalize(at - eye)
        v = torch.cross(up, view_vector)
        c = (up * view_vector).sum(-1)

        zeros = torch.zeros_like(v[:, 0])
        v_sk_sym = torch.stack([
            torch.stack([zeros, -v[:, 2], v[:, 1]], -1),
            torch.stack([v[:, 2], zeros, -v[:, 0]], -1),
            torch.stack([-v[:, 1], v[:, 0], zeros], -1)
        ], -2)
        I = torch.eye(3, device=eye.device)[None].repeat(batch_size, 1, 1)
        I_deg = -I.clone(); I_deg[:, 1, 1] = -I_deg[:, 1, 1]  # Degenerate case, when c is close to -1

        return torch.where(
            (c + 1) < 1e-5, I_deg,
            (I + v_sk_sym + (v_sk_sym @ v_sk_sym) / (1 + c).clamp(min=1e-13)).permute(0, 2, 1)
        )

    def get_dists(self, eye, rays, dist_model, one_shot_model):
        """
        Computes ray lengths upon hit
        :param eye: camera positions [B, N, 3]
        :param rays: normalized ray directions [B, N, 3]
        :param dist_model: distance model, callable which accepts query points
            in 3D space and outputs signed distances in them
        :param one_shot_model: model for one shot ray length estimation
        :return: ray lengths, hit positions, hit mask
        """
        eye = eye.expand_as(rays)

        origins, out_mask = one_shot_model(rays, eye) if one_shot_model is not None else (eye, torch.zeros(
            eye.shape[0], eye.shape[1], dtype=torch.bool, device=eye.device
        ))
        active_rays, active_origins = rays[~out_mask], origins[~out_mask]

        t = torch.linspace(0, self.iterations, self.iterations + 1, device=eye.device)
        func = OdeintFunc(dist_model, self.rendering_step, active_rays)

        if isinstance(dist_model, nn.Module):
            trajectories = odeint_adjoint(func, active_origins, t, method='euler')
        else:
            warn('dist_model is not an instance of nn.Module, falling back to standard backpropagation')
            trajectories = odeint(func, active_origins, t, method='euler')

        dists = (trajectories[-1] - trajectories[-2]).pow(2).sum(-1).sqrt()
        ray_lengths = (trajectories[-1] - eye[~out_mask]).pow(2).sum(-1).sqrt()

        hit_mask = torch.ones_like(dists).bool() if self.hit_thresh is None else dists.abs() < self.hit_thresh

        full_hit_mask = torch.zeros_like(out_mask)
        full_hit_mask[~out_mask] = hit_mask

        full_hit_points = torch.zeros_like(rays)
        full_hit_points[~out_mask] = trajectories[-1]

        full_ray_lengths = torch.full((rays.shape[0], rays.shape[1]), self.far, device=eye.device)
        full_ray_lengths[~out_mask] = ray_lengths

        return full_ray_lengths, full_hit_points, full_hit_mask

    def forward(self, eye, cam_transform, dist_model, texture_model, one_shot_model):
        """
        Renders
        :param eye: camera positions [B, N, 3]
        :param cam_transform: camera transformation matrix [3, 3]
        :param dist_model: distance model, callable which accepts query points
            in 3D space and outputs signed distances in them
        :param texture_model: texture model, callable; accepts points on the surface - outputs RGB values in
            them as well as the specular powers
        :param one_shot_model: model for one shot ray length estimation
        :return: image tensors with rendered objects
        """
        rays = self.rays[None].repeat(self.bsize, 1, 1) @ cam_transform

        ray_lengths, end_points, hit_mask = self.get_dists(eye, rays, dist_model, one_shot_model)

        hit_mask_float = hit_mask[:, :, None].repeat(1, 1, 3).float()

        image = self._background_color.expand_as(end_points).contiguous()
        surface_coords = end_points[hit_mask].view(self.bsize, -1, 3)

        if surface_coords.numel() == 0:
            return image.view(self.bsize, self.image_size, self.image_size, 3).permute(0, 3, 1, 2).contiguous()

        textures, specular_intesities = texture_model(surface_coords)

        texture_image = torch.zeros_like(image)
        texture_image[hit_mask] = textures
        image = image * (1 - hit_mask_float) + texture_image

        if hit_mask.any():
            if self._shading in ['distance', 'distance_inverted']:
                shader = ray_lengths.clamp(self.near, self.far)
                shader[~hit_mask] = shader[hit_mask].max()

                shader = shader - shader.view(self.bsize, -1).min(1)[0]
                shader = shader / shader.view(self.bsize, -1).max(1)[0].clamp(min=1e-7)
                if self._shading == 'distance_inverted':
                    shader = 1 - shader
                shader = shader[:, :, None]

            else:
                shader = torch.ones_like(image)

            image = image * (1 - hit_mask_float) + (image * shader).clamp(0, 1) * hit_mask_float

        image = image.view(self.bsize, self.image_size, self.image_size, 3)
        return image.permute(0, 3, 1, 2).contiguous()

    def render_look_at(self, dist_model, texture_model, one_shot_model=None, eye=None, at=(0, 0, 0)):
        """
        Look at rendering wrapper
        :param dist_model: distance model, callable which accepts query points
            in 3D space and outputs signed distances in them
        :param texture_model: texture model, callable; accepts points on the surface - outputs RGB values in
            them as well as the specular powers
        :param one_shot_model: model for one shot ray length estimation
        :param eye: camera positions [B, N, 3]
        :param at: point to look at
        :return: image tensor
        """
        if eye is None:
            eye = self.eye[None].repeat(self.bsize, 1, 1)

        cam_transform = self.look_at(eye, at)
        return self.forward(eye, cam_transform, dist_model, texture_model, one_shot_model)
