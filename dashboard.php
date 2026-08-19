<?php

require_once __DIR__ . '/includes/bootstrap.php';
require_login('client');

$user = current_user();

$stmt = db()->prepare(
    'SELECT p.id, p.title, p.description, p.status, p.progress,
            p.updated_at,
            pu.message AS latest_message,
            pu.created_at AS latest_update_time
     FROM projects p
     LEFT JOIN project_updates pu ON pu.id = (
         SELECT pu2.id FROM project_updates pu2
         WHERE pu2.project_id = p.id
         ORDER BY pu2.created_at DESC, pu2.id DESC
         LIMIT 1
     )
     WHERE p.client_id = ?
     ORDER BY p.updated_at DESC, p.id DESC'
);
$stmt->execute([$user['id']]);
$projects = $stmt->fetchAll();

require __DIR__ . '/includes/header.php';
?>
<h1 class="h3 mb-4">Welcome, <?= e($user['name']) ?></h1>

<?php if (!$projects): ?>
    <div class="alert alert-info">No project assigned yet. Please check back soon.</div>
<?php else: ?>
    <div class="row g-4">
        <?php foreach ($projects as $project): ?>
            <div class="col-12">
                <div class="card shadow-sm">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                            <h2 class="h5 mb-0"><?= e($project['title']) ?></h2>
                            <span class="badge text-bg-primary"><?= e($project['status']) ?></span>
                        </div>
                        <p class="text-muted mb-3"><?= e((string) $project['description']) ?></p>
                        <div class="mb-2">Progress: <strong><?= (int) $project['progress'] ?>%</strong></div>
                        <div class="progress mb-3" role="progressbar" aria-valuenow="<?= (int) $project['progress'] ?>" aria-valuemin="0" aria-valuemax="100">
                            <div class="progress-bar" style="width: <?= (int) $project['progress'] ?>%"></div>
                        </div>
                        <p class="mb-1"><strong>Latest Update:</strong> <?= e($project['latest_message'] ?: 'No updates yet.') ?></p>
                        <small class="text-muted">Updated: <?= e($project['latest_update_time'] ?: $project['updated_at']) ?></small>
                    </div>
                </div>
            </div>
        <?php endforeach; ?>
    </div>
<?php endif; ?>

<?php require __DIR__ . '/includes/footer.php'; ?>
